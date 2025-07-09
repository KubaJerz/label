from time import sleep
import uuid
import threading
import time
import uuid
import json
import torch
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu
from torch.utils.data import DataLoader, TensorDataset
from app.repositories.session_repository import SessionRepository

class ModelService:
    def __init__(self, session_repository=None):
        self.session_repo: SessionRepository = session_repository
        self.scoring_status = {}  # Track scoring operations
        self.models = {}

        self._load_models()

    def list_models(self):
        """List all available models"""
        return [
            {
                'name': model_dic['name'],
                'description':  model_dic['description']
            }
            for model_dic in self.models.values()
        ]
    
    def score_session_async(self, project_path, session_name, session_id, model_name='PuffSegmentation'):
        """Start async scoring of a session"""
        # Generate unique scoring ID
        scoring_id = str(uuid.uuid4())

        if model_name not in self.models:
            raise ValueError(f"model {model_name} not found \nAvailable models are:{self.list_models}")

        # Initialize status tracking
        self.scoring_status[scoring_id] = {
            'status': 'running',
            'session_id': session_id,
            'session_name': session_name,
            'model_name': model_name,
            'start_time': time.time(),
            'error': None
        }
        # Start async processing in a separate thread
        scoring_thread = threading.Thread(
            target=self._score_session_worker,  
            args=(scoring_id, project_path, session_name, session_id, model_name)
        )
        scoring_thread.daemon = True
        scoring_thread.start()
        
        return scoring_id
    
    def _load_models(self):
        try:
          # load smoking cnn
            smoking_cnn = SmokingCNN(window_size=3000, num_features=3)
            if os.path.exists('model.pth'): #hard coeded needs to be fixed
                smoking_cnn.load_state_dict(torch.load('model.pth', map_location='cpu'))
                smoking_cnn.eval()
                self.models['SmokingCNN'] = {
                    'model': smoking_cnn,
                    'name': 'SmokingCNN',
                    'description': 'smoking detection model simple cnn'
                }
            
            # load puff seg model
            puff_model = PuffSegmentation(in_channels=3, out_channels=1)
            if os.path.exists('puff_model.pt'): #hard coded needs to be fixed
                puff_model.load_state_dict(torch.load('puff_model.pt', map_location='cpu'))
                puff_model.eval()
                self.models['PuffSegmentation'] = {
                    'model': puff_model,
                    'name': 'PuffSegmentation',
                    'description': 'puff segmentation unet model'
                }
            
        except Exception as e:
            print(f"Error loading models: {e}")

    def _score_session_worker(self, scoring_id, project_path, session_name, session_id, model_name):
        try:
            print(f"Starting scoring for session {scoring_id}")

            df = pd.read_csv(f"{project_path}/{session_name}/accelerometer_data.csv")
            sample_interval = df['ns_since_reboot'].diff().median() * 1e-9
            sample_rate = 1 / sample_interval
            print(f"Sample rate: {sample_rate} Hz")

            # get model info
            model_info = self.models[model_name]
            model = model_info['model']

            # handel pred for each model
            if model_name == 'SmokingCNN':
                df = self._run_smoking_cnn(df, model)
            elif model_name == 'PuffSegmentation':
                df = self._run_puff_segmentation(df, model)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            #from preds in time domain to bouts
            new_bouts = self._extract_bouts_from_predictions(df, model_name)

            bouts = self.session_repo.get_bouts_by_session(session_id)
            json_bouts = json.loads(bouts) if bouts else []
            print(json_bouts + new_bouts)

            self.session_repo.set_bouts_by_session(session_id, json.dumps(json_bouts + new_bouts))

            # Update status on completion
            self.scoring_status[scoring_id].update({
                'status': 'completed',
                'end_time': time.time(),
                'bouts_count': len(new_bouts)
            })
        except Exception as e:
            print(f"Error during scoring: {e}")
            # Handle error (e.g., log it, update status, etc.)
            print(f"Error scoring session {scoring_id}: {e}")
            self.scoring_status[scoring_id].update({
                'status': 'error',
                'error': str(e),
                'end_time': time.time()
            })

        return new_bouts
    
    def _run_smoking_cnn(self, df, model):
        """run the original smoking cnn model"""
        fs = 50
        window_size_seconds = 60
        window_stride_seconds = 60

        X = []
        data = torch.tensor(df[['accel_x', 'accel_y', 'accel_z']].values, dtype=torch.float32)
        window_size = fs * window_size_seconds
        window_stride = fs * window_stride_seconds
        windowed_data = data.unfold(dimension=0, size=window_size, step=window_stride)
        X.append(windowed_data)

        X = torch.cat(X)

        with torch.no_grad():
            y_pred = model(X).sigmoid().cpu()
            y_pred = y_pred > 0.6
            y_pred = y_pred.numpy().flatten()
            y_pred = y_pred.repeat(3000)

        if len(y_pred) < len(df):
            y_pred = torch.cat([torch.tensor(y_pred), torch.zeros(len(df) - len(y_pred))])
        df['y_pred'] = y_pred * 20

        return df

    def _run_puff_segmentation(self, df, model):
        """run the puff segmentation unet model"""
        fs = 50
        window_size = 256  # your window size
        window_stride = 256  # stride = window size (no overlap)

        # prepare data
        data = torch.tensor(df[['accel_x', 'accel_y', 'accel_z']].values, dtype=torch.float32).T  # (3, seq_len)
        # create sliding windows
        if data.shape[1] < window_size:
            # pad if too short
            padding = window_size - data.shape[1]
            data = torch.nn.functional.pad(data, (0, padding))
        
        # create windows: unfold creates (channels, n_windows, window_size)
        windowed_data = data.unfold(dimension=1, size=window_size, step=window_stride)
        windowed_data = windowed_data.permute(1, 0, 2)  # (n_windows, channels, window_size)
        
        print(f"data with shape  {windowed_data.shape} ")

        with torch.no_grad():
            y_pred = model(windowed_data).sigmoid().cpu()  # (n_windows, window_size)
            print(torch.sum(y_pred > 0.5))
            y_pred = y_pred > 0.5  # threshold
            
            # reconstruct full timeline
            full_pred = torch.zeros(len(df))
            for i, window_pred in enumerate(y_pred):
                start_idx = i * window_stride
                end_idx = min(start_idx + window_size, len(df))
                actual_window_size = end_idx - start_idx
                full_pred[start_idx:end_idx] = window_pred[:actual_window_size]

        df['y_pred'] = full_pred.numpy() * 20

        return df

    

    def _extract_bouts_from_predictions(self, df, model_name):
        """extract bouts from prediction timeline"""
        smoking_bouts = []
        current_bout = None
        
        for i in range(len(df)):
            if df['y_pred'].iloc[i] > 0:
                if current_bout is None:
                    current_bout = [int(df['ns_since_reboot'].iloc[i]), None]
                # else:
                current_bout[1] = int(df['ns_since_reboot'].iloc[i])
            else:
                if current_bout is not None:
                    smoking_bouts.append(current_bout)
                    current_bout = None

        # remove bouts shorter than 20 seconds is ccn model and format as dictionaries
        label = f"{model_name}"
        
        smoking_bouts = [
            {
                'start': bout[0],
                'end': bout[1], 
                'label': label
            }
            for bout in smoking_bouts 
            if ((bout[1] - bout[0]) >= 2 * 1e9)
        ]
        
        print(f"generated {len(smoking_bouts)} bouts with label: {label} in _extract_bouts_from_predictions")
        return smoking_bouts
    
    def get_scoring_status(self, scoring_id):
        """Get the status of a scoring operation"""
        return self.scoring_status.get(scoring_id, {'status': 'not_found'})
    
class SmokingCNN(nn.Module):
    def __init__(self, window_size=100, num_features=6):
        super(SmokingCNN, self).__init__()
        kernel_size = 3

        self.c1 = nn.Conv1d(in_channels=num_features, out_channels=4, kernel_size=kernel_size)
        self.c2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=kernel_size)
        self.c3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=kernel_size)

        self.classifier = nn.Linear(16, 1)
    
    
    def forward(self, x):
        x = self.c1(x)
        x = relu(x)
        x = self.c2(x)
        x = relu(x)
        x = self.c3(x)
        x = relu(x)
        x = x.mean(dim=2)

        x = self.classifier(x)
        return x

class DoubleConv(nn.Module):
    """(Conv1d -> ReLU -> Conv1d -> ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class PuffSegmentation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final output
        self.out = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        # Bottleneck
        x4 = self.bottleneck(self.pool3(x3))

        # Decoder
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.out(x).squeeze(dim=1)
