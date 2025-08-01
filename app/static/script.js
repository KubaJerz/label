import * as eventListeners from './eventListeners.js';
import { ensureSessionBoutsIsArray, generateDefaultColor, pixelToData, createDarkLayout, createLightLayout } from './js/helpers.js'
import ProjectAPI from './js/api/projectAPI.js';
import ProjectController from './js/controllers/projectController.js';
import { 
    updateCurrentProjectPill, 
    resetScoreButton, 
    showCreateProjectForm, 
    showTableView, 
    showNotification,
    showBulkUploadForm,
    hideModal,
    resetForm
} from './js/ui/uiUtils.js';
import SessionAPI from './js/api/sessionAPI.js';
import SessionService from './js/services/sessionService.js';
import SessionController from './js/controllers/sessionController.js';
import { ActionButtonTemplates, ActionButtonHandlers } from './js/templates/actionButtonTemplates.js';
import { SessionListTemplates, SessionListHandlers } from './js/templates/sessionListTemplates.js';

function checkUrlParameters() {
    const urlParams = new URLSearchParams(window.location.search);
    const participantCode = urlParams.get('participant');
    const createProject = urlParams.get('create_project');
    const projectId = urlParams.get('project_id');
    
    if (participantCode && createProject === 'true') {
        // Pre-fill participant code and show create project modal
        setTimeout(() => {
            document.getElementById('project-participant').value = participantCode;
            showCreateProjectForm();
        }, 500); // Small delay to ensure page is loaded
    }
    
    if (projectId) {
        // Set the current project and fetch its sessions
        currentProjectId = parseInt(projectId);
        // Store in sessionStorage for persistence across page navigation
        sessionStorage.setItem('currentProjectId', currentProjectId.toString());
        setTimeout(async () => {
            try {
                // Fetch project details to get the name
                const projects = await ProjectAPI.fetchProjects();
                const project = projects.find(p => p.project_id === currentProjectId);
                
                if (project) {
                    // Update UI to show this project is selected
                    updateCurrentProjectPill(project.project_name);
                    
                    // Update active state in dropdown if it exists
                    const dropdownItems = document.querySelectorAll('#project-dropdown-menu .dropdown-item');
                    dropdownItems.forEach(item => {
                        item.classList.remove('active');
                        item.removeAttribute('aria-current');
                        if (parseInt(item.dataset.projectId) === currentProjectId) {
                            item.classList.add('active');
                            item.setAttribute('aria-current', 'page');
                        }
                    });
                    
                    // Fetch and display sessions for this project
                    await ProjectController.fetchProjectSessions(currentProjectId);
                }
            } catch (error) {
                console.error('Error setting up project from URL:', error);
            }
        }, 500);
    }
}


/*
OTHER
*/


// Update the sessions list in the UI
function updateSessionsList() {
    const sessionList = document.getElementById("session-list");
    const tbody = document.getElementById("sessions-table-body");
    
    // Check if required elements exist
    if (!sessionList && !tbody) {
        console.warn('Neither session-list nor sessions-table-body elements found on this page');
        return;
    }
    
    // Clear existing content
    if (sessionList) {
        sessionList.innerHTML = "";
    }
    if (tbody) {
        tbody.innerHTML = "";
    }
    
    if (sessions.length === 0) {
        // Display a message for empty sessions
        if (tbody) {
            tbody.innerHTML = SessionListTemplates.emptyState();
        }
        return;
    }
    
    // Populate sessions
    sessions.forEach(session => {
        if (session.keep == 0) return; // Skip discarded sessions
        
        // Sidebar entry (only if sessionList exists)
        if (sessionList) {
            const li = document.createElement("li");
            li.className = "nav-item";
            li.innerHTML = SessionListTemplates.sidebarItem(session, currentActiveSession);
            sessionList.appendChild(li);
        }

        // Table row (only if tbody exists)
        if (tbody) {
            const row = document.createElement("tr");
            const sessionId = session.session_id;
            
            // Use template for table row
            row.innerHTML = SessionListTemplates.tableRow(session);
            tbody.appendChild(row);

            // Setup event handlers using the template handlers
            SessionListHandlers.setupTableRowHandlers(
                sessionId, 
                decideSession, 
                toggleVerifiedStatus
            );

        }
    });
}
async function pollScoringStatus(scoringId, sessionId, sessionName, deviceType = 'cpu') {
    const maxPolls = 120; // 2 minutes max
    let pollCount = 0;
    const deviceLabel = deviceType.toUpperCase();
    
    console.log(`Starting ${deviceLabel} scoring status polling for ${scoringId}`);
    
    const poll = setInterval(async () => {
        try {
            console.log(`Polling attempt ${pollCount} for ${deviceLabel} scoring_id:`, scoringId); 
            const response = await fetch(`/api/scoring_status/${scoringId}`);
            console.log(`${deviceLabel} polling response status:`, response.status); 
            const status = await response.json();
            
            pollCount++;
            
            if (status.status === 'completed') {
                clearInterval(poll);
                
                // Create device-specific success message
                const deviceInfo = status.device_used ? ` (${status.device_used})` : ` on ${deviceLabel}`;
                const boutsMessage = status.bouts_count ? ` Found ${status.bouts_count} bouts.` : '';
                
                showNotification(
                    `${deviceLabel} scoring complete for ${sessionName}${deviceInfo}!${boutsMessage}`, 
                    'success'
                );
                resetScoreButton(sessionId);
                
                // Force refresh the session data from the server
                const sessionResponse = await fetch(`/api/session/${sessionId}`);
                if (sessionResponse.ok) {
                    const sessionData = await sessionResponse.json();
                    
                    // Update the session in our local sessions array
                    const sessionIndex = sessions.findIndex(s => s.session_id == sessionId);
                    if (sessionIndex !== -1) {
                        // Parse bouts if they're a string
                        let bouts = sessionData.bouts;
                        if (typeof bouts === 'string') {
                            try {
                                bouts = JSON.parse(bouts);
                            } catch (e) {
                                console.error('Error parsing bouts:', e);
                                bouts = [];
                            }
                        }
                        sessions[sessionIndex].bouts = bouts || [];
                        sessions[sessionIndex].data = sessionData.data;
                        
                        // Extract the labeling name from the new bouts and create/update labeling
                        if (bouts && bouts.length > 0) {
                            // Get the newest bout's label (just the model name, no device info)
                            const newestBout = bouts[bouts.length - 1];
                            const modelLabelingName = newestBout.label;
                            
                            console.log(`Creating/updating labeling: ${modelLabelingName}`);
                            
                            if (modelLabelingName && currentProjectId) {
                                await ProjectController.createOrUpdateModelLabeling(modelLabelingName);
                                // Automatically select the new model labeling to show the results
                                selectLabeling(modelLabelingName);
                            }
                        }
                    }
                }
                
                // If this session is currently being visualized, refresh it
                if (currentSessionId == sessionId) {
                    console.log(`Refreshing currently visualized session with new ${deviceLabel} bouts`);
                    const plotDiv = document.getElementById('timeSeriesPlot');
                    let viewState = null;
                    if (plotDiv && plotDiv._fullLayout && plotDiv._fullLayout.xaxis && plotDiv._fullLayout.yaxis) {
                        viewState = {
                            xrange: plotDiv._fullLayout.xaxis.range.slice(),
                            yrange: plotDiv._fullLayout.yaxis.range.slice()
                        };
                    }
                    visualizeSession(sessionId).then(() => {
                        if (viewState && document.getElementById('timeSeriesPlot')) {
                            Plotly.relayout('timeSeriesPlot', {
                                'xaxis.range': viewState.xrange,
                                'yaxis.range': viewState.yrange
                            });
                        }
                    });
                }
                
            } else if (status.status === 'error') {
                clearInterval(poll);
                const errorMsg = status.error || 'Unknown error occurred';
                showNotification(`${deviceLabel} scoring failed: ${errorMsg}`, 'error');
                resetScoreButton(sessionId);
                
            } else if (pollCount >= maxPolls) {
                clearInterval(poll);
                showNotification(`${deviceLabel} scoring is taking longer than expected`, 'warning');
                resetScoreButton(sessionId);
            } else {
                // Update UI with current status if available
                if (status.status === 'running' && deviceType === 'gpu') {
                    console.log(`GPU scoring in progress... (${pollCount}/${maxPolls})`);
                }
            }
            
        } catch (error) {
            console.error(`Error polling ${deviceLabel} scoring status:`, error);
            clearInterval(poll);
            showNotification(`Failed to get ${deviceLabel} scoring status`, 'error');
            resetScoreButton(sessionId);
        }
    }, 1000); // Poll every second
}



// Show visualization view
async function visualizeSession(sessionId) {
    // If we're already viewing a session and switching to another, save changes first
    if (currentSessionId && currentSessionId !== sessionId) {
        const currentSession = sessions.find(s => s.session_id == currentSessionId);
        if (currentSession) {
            try {
                console.log(`Saving bout changes before switching from ${currentSessionId} to ${sessionId}`);
                await SessionAPI.updateSessionMetadata(currentSession);
            } catch (error) {
                console.error('Error saving bout changes before switching sessions:', error);
            }
        }
    }
    // Clean up previous event handlers
    activeHandlers.forEach(h => {
        document.removeEventListener(h.type, h.handler);
    });
    activeHandlers.length = 0;
    
    // Clean up overlays using overlay manager
    window.OverlayManager.cleanupOverlays();
    
    // Find the session by ID
    const session = sessions.find(s => s.session_id == sessionId);
    if (!session) {
        console.error('Session not found:', sessionId);
        return;
    }
    
    // Check if the session is actually available (not deleted)
    if (!SessionService.isSessionAvailable(session)) {
        console.error('Attempted to visualize deleted session:', sessionId);
        
        // Find an available session to visualize instead
        const availableSessions = SessionService.getFilteredSessions(sessions);
        if (availableSessions.length > 0) {
            console.log('Redirecting to first available session');
            visualizeSession(availableSessions[0].session_id);
        } else {
            console.log('No available sessions');
            showTableView();
        }
        return;
    }
    
    // Set the current session name/id
    currentSessionId = sessionId;
    currentActiveSession = session.session_name;
    
    dragContext.currentSession = session;

    if (!session.data || session.data.length === 0) {
        const { bouts, data } = await SessionAPI.loadSessionData(sessionId);
        session.bouts = bouts;
        session.data = data;
        if (!session.data || session.data.length === 0) {
            console.error('No valid data for session:', currentActiveSession);
            return;
        }
    }
    const validData = SessionService.validateSessionData(session);
    if (!validData) {
        console.error('Invalid data format:', session.data);
        return;
    }

    document.getElementById("table-view").style.display = "none";
    document.getElementById("visualization-view").style.display = "flex";

    // Update sidebar highlighting for the current session
    updateSidebarHighlighting();

    if (session.status === "Initial") {
        session.status = "Visualized";
        await SessionAPI.updateSessionMetadata(session);
        updateSessionsList();
    }

    // If currentLabelingName is "No Labeling", set it to the first available labeling
    if (currentLabelingName === "No Labeling") {
        const availableLabelings = Object.keys(session.labelings || {});
        console.log('Available labelings:', availableLabelings);
        if (availableLabelings.length > 0) {
            currentLabelingName = availableLabelings[0];
        }
    }
    console.log('Current selected labeling:', currentLabelingName);
    const actionButtons = document.getElementById("action-buttons");
    actionButtons.innerHTML = "";

    function toggleDarkMode(){
        const plotDiv = document.getElementById('timeSeriesPlot');
        if (!plotDiv) return;

        Plotly.relayout(plotDiv, {
            'xaxis.autorange': false,
            'yaxis.autorange': false
        });

        // Get current zoom ranges before changing layout
        const viewState = {
            xrange: plotDiv.layout?.xaxis?.range?.slice() || plotDiv._fullLayout?.xaxis?.range?.slice(),
            yrange: plotDiv.layout?.yaxis?.range?.slice() || plotDiv._fullLayout?.yaxis?.range?.slice(),
        };

        const body = document.body;
        const isDark = body.classList.toggle('dark-mode');
        localStorage.setItem('dark-mode', isDark);

        // Switch the icon
        const icon = document.getElementById('darkModeIcon');
        if (icon) {
            icon.classList.remove('fa-moon', 'fa-sun');
            icon.classList.add(isDark ? 'fa-sun' : 'fa-moon');
        }

        const layout = isDark ? darkLayout : lightLayout;
        if (viewState.xrange && viewState.yrange) {
            layout.xaxis.range = viewState.xrange;
            layout.yaxis.range = viewState.yrange;
        }
        Plotly.react('timeSeriesPlot', traces, layout, config);
    }

    // Use template for action buttons
    actionButtons.innerHTML = ActionButtonTemplates.visualizationActionButtons({
        isSplitting: isSplitting,
        isVerified: session.verified,
        labelingName: currentLabelingName,
        labelingColor: currentLabelingJSON ? currentLabelingJSON.color : '#000000',
    });
    
    // Setup event listeners using the template handlers
    ActionButtonHandlers.setupVisualizationButtons({
        onDeleteBouts: () => deleteAllBouts(),
        onDelete: () => decideSession(currentSessionId, false),
        onVerify: () => toggleVerifiedStatus(),
        onSplit: () => toggleSplitMode(),
        onScore: () => SessionController.scoreSession(currentSessionId, session.project_name, session.session_name),
        onDarkMode: () => toggleDarkMode(),
        isSplitting: isSplitting
    });

    // Show correct dark mode icon
    const icon = document.getElementById('darkModeIcon');
    const isDark = document.body.classList.contains('dark-mode');
    if (icon) {
    icon.classList.remove('fa-moon', 'fa-sun');
    icon.classList.add(isDark ? 'fa-sun' : 'fa-moon');
    }

    const dataToPlot = session.data;
    if (!dataToPlot || dataToPlot.length === 0) {
        console.error('No data to plot for session:', sessionName);
        return;
    }

    const timestamps = dataToPlot.map(d => d.ns_since_reboot).filter(t => t);
    const xValues = dataToPlot.map(d => d.accel_x).filter(accel_x => typeof accel_x === 'number');
    const yValues = dataToPlot.map(d => d.accel_y).filter(accel_y => typeof accel_y === 'number');
    const zValues = dataToPlot.map(d => d.accel_z).filter(accel_z => typeof accel_z === 'number');
    const labels = dataToPlot.map(d => d.label).filter(label => typeof label === 'number');
    if (timestamps.length === 0) {
        console.error('No valid timestamps for plotting');
        return;
    }

    // With this more efficient approach:
    minTimestamp = timestamps.reduce((min, val) => Math.min(min, val), Infinity);

    // And similarly for maxTimestamp:
    maxTimestamp = timestamps.reduce((max, val) => Math.max(max, val), -Infinity);

    const traces = [
        { x: timestamps, y: xValues, name: 'X Axis', type: 'scatter', mode: 'lines', line: { color: '#17BECF' } },
        { x: timestamps, y: yValues, name: 'Y Axis', type: 'scatter', mode: 'lines', line: { color: '#FF7F0E' } },
        { x: timestamps, y: zValues, name: 'Z Axis', type: 'scatter', mode: 'lines', line: { color: '#2CA02C' } },
        { x: timestamps, y: labels, name: 'Z Axis', type: 'scatter', mode: 'lines', line: { color: '#FF0000' } }
    ];
    const shapes = [];

    splitPoints.forEach(point => {
        shapes.push({
            type: 'line',
            x0: point,
            x1: point,
            y0: 0,
            y1: 1,
            yref: 'paper', // Span full y-axis height
            line: {
                color: '#FF0000', // Grey
                width: 2,
                dash: 'dash'
            }
        });
    });

    const darkLayout = createDarkLayout(shapes);
    const lightLayout = createLightLayout(shapes);
    const isDarkMode = document.body.classList.contains('dark-mode');
    const layout = isDarkMode ? darkLayout : lightLayout; 

    const container = document.querySelector('.plot-container');
    container.querySelectorAll('.drag-overlay').forEach(el => el.remove());

    // Disables double-click zoom
    const config = {
        doubleClick: false
    };
    
    Plotly.newPlot('timeSeriesPlot', traces, layout, config).then(() => {
        const plotDiv = document.getElementById('timeSeriesPlot');

        ensureSessionBoutsIsArray(session);

        const overlays = session.bouts.map((bout, index) => createBoutOverlays(index, container));
        // Update all overlay positions using overlay manager
        window.OverlayManager.updateOverlaysForLabelingChange(session, currentLabelingName);

        // Handle plot click for splitting
        plotDiv.on('plotly_click', function(data) {
            if (isSplitting) {
                const splitPoint = data.points[0].x;
                if (!splitPoints.includes(splitPoint)) {
                    // Get current zoom level before adding split point
                    let viewState = null;
                    if (plotDiv && plotDiv._fullLayout && plotDiv._fullLayout.xaxis) {
                        viewState = {
                            xrange: plotDiv._fullLayout.xaxis.range.slice(),
                            yrange: plotDiv._fullLayout.yaxis.range.slice()
                        };
                    }
                    
                    splitPoints.push(splitPoint);
                    
                    // Refresh plot to show new split point marker and restore zoom level
                    visualizeSession(sessionId).then(() => {
                        if (viewState && plotDiv) {
                            Plotly.relayout(plotDiv, {
                                'xaxis.range': viewState.xrange,
                                'yaxis.range': viewState.yrange
                            });
                        }
                    });
                }
            }
        });
        // Update overlays on plot relayout (pan, zoom, etc.)
        plotDiv.on('plotly_relayout', () => {
            console.log('Plotly relayout event');
            window.OverlayManager.updateOverlaysForLabelingChange(session, currentLabelingName);
        });
        // Update overlays during pan/zoom interaction
        plotDiv.on('plotly_relayouting', () => {
            window.OverlayManager.updateOverlaysForLabelingChange(session, currentLabelingName);
        });
        // Update overlays on window resize
        window.addEventListener('resize', () => {
            Plotly.Plots.resize(plotDiv).then(() => {
                window.OverlayManager.updateOverlaysForLabelingChange(session, currentLabelingName);
            });
        });         
        // Handle double click for pan and zoom
        plotDiv.on('plotly_doubleclick', (eventData) => {
            const mouseMode = plotDiv._fullLayout.dragmode;
            if (mouseMode === 'pan') { 
                createNewBout();
            }
            else if (mouseMode === 'zoom') {
                Plotly.relayout(plotDiv, {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
            }
        });
    });


}
// When opening the modal
const labelingModal = document.getElementById('labelingModal');
if (labelingModal) {
    labelingModal.addEventListener('shown.bs.modal', function() {
        console.log('Labeling modal opened');
        // Fetch and display labelings when the modal is opened
        ProjectController.fetchAndDisplayLabelings(currentProjectId);
    });
}


// You can also initialize it in the DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the global variable
    labelingsList = document.getElementById('available-labelings-list');
    
    if (labelingsList) {
        // Handle clicks on the plus icon
        labelingsList.addEventListener('click', function(event) {
            if (event.target.classList.contains('fa-plus')) {
                event.preventDefault();
                ProjectController.createNewLabeling();
            }
        });
        
        // Handle hover effects on the plus icon
        labelingsList.addEventListener('mouseenter', function(event) {
            if (event.target.classList.contains('fa-plus')) {
                event.target.style.background = 'rgba(0,0,0,0.1)';
                event.target.style.borderRadius = '50%';
                event.target.style.padding = '4px';
                event.target.style.cursor = 'pointer';
            }
        }, true);
        
        labelingsList.addEventListener('mouseleave', function(event) {
            if (event.target.classList.contains('fa-plus')) {
                event.target.style.background = 'rgba(224,224,224,0)';
            }
        }, true);
    }

});


function getLabelingFromLabelingName(labelingName) {
    if (!labelings || labelings.length === 0) {
        console.warn('No labelings available to search from');
        return null;
    }
    const labeling = labelings.find(l => l.name === labelingName);
    if (!labeling) {
        console.warn(`Labeling not found for name: ${labelingName}`);
        return null;
    }
    return labeling;
}

function selectLabeling(labelingName) {
    console.log(`Selecting labeling: ${labelingName}`);
    currentLabelingName = labelingName;
    currentLabelingJSON = getLabelingFromLabelingName(labelingName);

    // Update the current labeling name in visualization view with color and maintain interactivity
    const currentLabelingNameElement = document.getElementById('current-labeling-name');
    if (currentLabelingNameElement) {

        const currentLabeling = getLabelingFromLabelingName(labelingName);
        const labelingColor = currentLabeling.color
        
        currentLabelingNameElement.innerHTML = `
            <div class="color-circle me-1" style="width: 12px; height: 12px; border-radius: 50%; background-color: ${labelingColor}; border: 1px solid #ccc; display: inline-block;"></div>
            ${labelingName}
        `;
        
        // Maintain the cursor pointer and transition styles
        currentLabelingNameElement.style.cursor = 'pointer';
        currentLabelingNameElement.style.transition = 'background-color 0.2s ease, transform 0.1s ease';
    }
    
    // If we're in visualization view, update the overlays to show only bouts matching this labeling
    console.log(dragContext.currentSession);
    console.log(dragContext.currentSession.bouts);
    console.log(currentLabelingName)
    if (dragContext.currentSession && dragContext.currentSession.bouts) {
        window.OverlayManager.updateOverlaysForLabelingChange(dragContext.currentSession, currentLabelingName);
    }
    
    // You can add more logic here for what happens when a labeling is selected
}

function updateCurrentLabelingHeader(labelingName = null) {
    const displayName = labelingName || currentLabelingName;

    // Update visualization view name display
    const currentLabelingNameElement = document.getElementById('current-labeling-name');
    if (currentLabelingNameElement) {
        if (displayName && displayName !== 'No Labeling') {
            const currentLabeling = getLabelingFromLabelingName(displayName);
            const labelingColor = currentLabeling ? currentLabeling.color : generateDefaultColor(0);

            currentLabelingNameElement.innerHTML = `
                <div class="color-circle me-1" style="width: 12px; height: 12px; border-radius: 50%; background-color: ${labelingColor}; border: 1px solid #ccc; display: inline-block;"></div>
                ${displayName}
            `;
        } else {
            currentLabelingNameElement.textContent = 'No Labeling';
        }
        
        // Always maintain the interactive styling
        currentLabelingNameElement.style.cursor = 'pointer';
        currentLabelingNameElement.style.transition = 'background-color 0.2s ease, transform 0.1s ease';
    }
}
function createBoutOverlays(index, container) {
    const dragOverlay = document.createElement('div');
    dragOverlay.id = `drag-overlay-${index}`;
    dragOverlay.className = 'drag-overlay';
    dragOverlay.dataset.boutIndex = index; // Store index for reference
    container.appendChild(dragOverlay);
    
    const leftOverlay = document.createElement('div');
    leftOverlay.id = `left-overlay-${index}`;
    leftOverlay.className = 'left-overlay';
    leftOverlay.dataset.boutIndex = index;
    container.appendChild(leftOverlay);

    const rightOverlay = document.createElement('div');
    rightOverlay.id = `right-overlay-${index}`;
    rightOverlay.className = 'right-overlay';
    rightOverlay.dataset.boutIndex = index;
    container.appendChild(rightOverlay);
    
    // Dragging state variables
    let isDragging = false;
    let isResizingLeft = false;
    let isResizingRight = false;
    let startX = 0;
    let originalLeft = 0;
    let originalWidth = 0;
    let hasMovedBout = false; // Track if bout was actually moved
    
    // Add double-click event to remove the bout
    dragOverlay.addEventListener('dblclick', function() {
        const boutIndex = parseInt(dragOverlay.dataset.boutIndex);
        if (dragContext.currentSession && dragContext.currentSession.bouts) {
            // Get current zoom level before removing bout
            const plotDiv = document.getElementById('timeSeriesPlot');
            let viewState = null;
            if (plotDiv && plotDiv._fullLayout && plotDiv._fullLayout.xaxis) {
                viewState = {
                    xrange: plotDiv._fullLayout.xaxis.range.slice(),
                    yrange: plotDiv._fullLayout.yaxis.range.slice()
                };
            }
            
            dragContext.currentSession.bouts.splice(boutIndex, 1);
            console.log(`Removed bout ${boutIndex}`);
            SessionAPI.updateSessionMetadata(dragContext.currentSession);
            
            // Refresh the plot and restore zoom level
            visualizeSession(currentSessionId).then(() => {
                if (viewState && plotDiv) {
                    Plotly.relayout(plotDiv, {
                        'xaxis.range': viewState.xrange,
                        'yaxis.range': viewState.yrange
                    });
                }
            });
        }
    });
    // Element-specific mouse events for dragging
    dragOverlay.addEventListener('mousedown', function(e) {
        isDragging = true;
        hasMovedBout = false; // Reset movement flag
        startX = e.clientX;
        originalLeft = parseInt(dragOverlay.style.left) || 0;
        originalWidth = parseInt(dragOverlay.style.width) || 0;
        e.preventDefault();
        
        // Add temporary handlers
        addTemporaryMouseHandlers();
    });
    
    // Mouse handlers for left and right resizing
    leftOverlay.addEventListener('mousedown', function(e) {
        isResizingLeft = true;
        hasMovedBout = false; // Reset movement flag
        startX = e.clientX;
        originalLeft = parseInt(dragOverlay.style.left) || 0;
        originalWidth = parseInt(dragOverlay.style.width) || 0;
        e.preventDefault();
        dragOverlay.style.cursor = 'w-resize';
        
        // Add temporary handlers
        addTemporaryMouseHandlers();
    });
    
    rightOverlay.addEventListener('mousedown', function(e) {
        isResizingRight = true;
        hasMovedBout = false; // Reset movement flag
        startX = e.clientX;
        originalLeft = parseInt(dragOverlay.style.left) || 0;
        originalWidth = parseInt(dragOverlay.style.width) || 0;
        e.preventDefault();
        dragOverlay.style.cursor = 'e-resize';
        
        // Add temporary handlers
        addTemporaryMouseHandlers();
    });
    // Add temporary mousemove and mouseup handlers that clean themselves up
    function addTemporaryMouseHandlers() {
        const mouseMoveHandler = function(e) {
            if (!isDragging && !isResizingLeft && !isResizingRight) return;
        
            const boutIndex = parseInt(dragOverlay.dataset.boutIndex);
            const plotDiv = document.getElementById('timeSeriesPlot');
            const xAxis = plotDiv._fullLayout.xaxis;
            
            // Regular dragging - move entire overlay
            if (isDragging) {
                const dx = e.clientX - startX;
                dragOverlay.style.left = `${originalLeft + dx}px`;
                
                // Mark that the bout has moved if there's significant movement
                if (Math.abs(dx) > 2) {
                    hasMovedBout = true;
                }
                
                // Convert pixel position back to data coordinates
                const newX0 = pixelToData(originalLeft + dx, xAxis);
                const newX1 = pixelToData(originalLeft + originalWidth + dx, xAxis);
                
                updateBoutData(boutIndex, newX0, newX1);
                
                // Update left and right handle positions
                leftOverlay.style.left = `${originalLeft + dx}px`;
                rightOverlay.style.left = `${originalLeft + originalWidth + dx - parseInt(rightOverlay.style.width || 10)}px`;
            }
            
            // Left resize - adjust left side of overlay
            else if (isResizingLeft) {
                const dx = e.clientX - startX;
                const newLeft = originalLeft + dx;
                const newWidth = originalWidth - dx;
                
                // Prevent width from becoming negative
                if (newWidth <= 10) return;
                
                // Mark that the bout has moved if there's significant movement
                if (Math.abs(dx) > 2) {
                    hasMovedBout = true;
                }
                
                dragOverlay.style.left = `${newLeft}px`;
                dragOverlay.style.width = `${newWidth}px`;
                leftOverlay.style.left = `${newLeft}px`;
                
                // Convert pixel position back to data coordinates
                const newX0 = pixelToData(newLeft, xAxis);
                const newX1 = pixelToData(newLeft + newWidth, xAxis);
                
                updateBoutData(boutIndex, newX0, newX1);
            }
            
            // Right resize - adjust right side of overlay
            else if (isResizingRight) {
                const dx = e.clientX - startX;
                const newWidth = originalWidth + dx;
                
                // Prevent width from becoming negative
                if (newWidth <= 10) return;
                
                // Mark that the bout has moved if there's significant movement
                if (Math.abs(dx) > 2) {
                    hasMovedBout = true;
                }
                
                dragOverlay.style.width = `${newWidth}px`;
                rightOverlay.style.left = `${originalLeft + newWidth - parseInt(rightOverlay.style.width || 10)}px`;
                
                // Convert pixel position back to data coordinates
                const newX0 = pixelToData(originalLeft, xAxis);
                const newX1 = pixelToData(originalLeft + newWidth, xAxis);
                
                updateBoutData(boutIndex, newX0, newX1);
            }
        };
        
        const mouseUpHandler = function() {
            isDragging = isResizingLeft = isResizingRight = false;
            if (dragOverlay) dragOverlay.style.cursor = 'move';

            // Only save bout changes if the bout was actually moved
            if (hasMovedBout) {
                saveBoutChanges().catch(err => console.error('Error in saveBoutChanges:', err));
            }
            
            // Remove the temporary handlers
            document.removeEventListener('mousemove', mouseMoveHandler);
            document.removeEventListener('mouseup', mouseUpHandler);
            
            // Also remove from our tracking array
            const moveIndex = activeHandlers.findIndex(h => h.handler === mouseMoveHandler);
            if (moveIndex !== -1) activeHandlers.splice(moveIndex, 1);
            
            const upIndex = activeHandlers.findIndex(h => h.handler === mouseUpHandler);
            if (upIndex !== -1) activeHandlers.splice(upIndex, 1);
        };
        
        // Store references to remove them later
        activeHandlers.push({ 
            type: 'mousemove', 
            handler: mouseMoveHandler 
        });
        activeHandlers.push({ 
            type: 'mouseup', 
            handler: mouseUpHandler 
        });
        
        document.addEventListener('mousemove', mouseMoveHandler);
        document.addEventListener('mouseup', mouseUpHandler);
    }
    
    // Helper function to update bout data
    function updateBoutData(boutIndex, x0, x1) {
        if (dragContext.currentSession && dragContext.currentSession.bouts) {
            if (dragContext.currentSession.bouts[boutIndex]) {
                dragContext.currentSession.bouts[boutIndex]['start'] = x0;
                dragContext.currentSession.bouts[boutIndex]['end'] = x1;
            } else {
                // If boutIndex is out of bounds, log an error
                console.error(`Bout index ${boutIndex} not found in session ${dragContext.currentSession.name}`);
            }
        } else {
            console.error("No current session in drag context");
        }
    }
    
    // Make saveBoutChanges async and ensure it completes
    async function saveBoutChanges() {
        if (sessions && sessions.length > 0) {
            const session = sessions.find(s => s.session_id == currentSessionId);
            if (session) {
                console.log(`Saving bout changes for session ${session.session_name} (ID: ${currentSessionId})`);
                try {
                    await SessionAPI.updateSessionMetadata(session);
                    console.log('Successfully saved bout changes to server');
                } catch (error) {
                    console.error('Failed to save bout changes:', error);
                    alert('Failed to save bout changes. Your changes may be lost when switching sessions.');
                }
            } else {
                console.error(`Session not found for saving: ID ${currentSessionId}`);
            }
        } else {
            console.error("No sessions available for saving");
        }
    }

    return { dragOverlay, leftOverlay, rightOverlay };
}


// Add this function to update the overlay positions
function updateOverlayPositions(plotDiv, bout, index) {
    // Ensure the plotDiv is available and initialized
    if (!plotDiv || !plotDiv._fullLayout || !plotDiv._fullLayout.xaxis) {
        console.error('Plot layout not available');
        return;
    }
    const bout_start = bout['start'];
    const bout_end = bout['end'];
    const bout_label = bout['label'];

    // Get the overlay elements
    const dragOverlay = document.getElementById(`drag-overlay-${index}`);
    const leftOverlay = document.getElementById(`left-overlay-${index}`);
    const rightOverlay = document.getElementById(`right-overlay-${index}`);
    console.log(`Updating overlays for bout ${index}: start=${bout_start}, end=${bout_end}, label=${bout_label}`);
    console.log(dragOverlay, leftOverlay, rightOverlay);
    if (!dragOverlay || !leftOverlay || !rightOverlay) return;

    // Only show and position overlays for the currently selected labeling
    console.log(`Checking bout label: ${bout_label} against current labeling: ${currentLabelingName}`);
    if (bout_label !== currentLabelingName) {
        hideOverlay(index);
        return;
    }

    // Show overlays (in case they were hidden)
    dragOverlay.style.display = 'block';
    leftOverlay.style.display = 'block';
    rightOverlay.style.display = 'block';

    // Get axis object from Plotly layout
    const xAxis = plotDiv._fullLayout.xaxis;
    const yAxis = plotDiv._fullLayout.yaxis;
    
    // Convert data coordinates to pixel positions
    const pixelX0 = xAxis._length * (bout_start - xAxis.range[0]) / (xAxis.range[1] - xAxis.range[0]) + xAxis._offset;
    const pixelX1 = xAxis._length * (bout_end - xAxis.range[0]) / (xAxis.range[1] - xAxis.range[0]) + xAxis._offset;
    
    // Set handle size
    const handleWidth = 20;
    const handleHeight = yAxis._length;
    
    const currentLabeling = getLabelingFromLabelingName(currentLabelingName);
    const labelingColor = currentLabeling ? currentLabeling.color : generateDefaultColor(0);
    console.log(currentLabeling)
    // Set main overlay position and size
    dragOverlay.style.position = 'absolute';
    dragOverlay.style.left = `${pixelX0}px`;
    dragOverlay.style.width = `${pixelX1 - pixelX0}px`;
    dragOverlay.style.top = `${yAxis._offset}px`;
    dragOverlay.style.height = `${handleHeight}px`;
    dragOverlay.style.backgroundColor = labelingColor + "77"
    dragOverlay.style.border = '2px solid black';
    
    // Set left handle position and size
    leftOverlay.style.position = 'absolute';
    leftOverlay.style.left = `${pixelX0}px`;
    leftOverlay.style.width = `${handleWidth}px`;
    leftOverlay.style.top = `${yAxis._offset}px`;
    leftOverlay.style.height = `${handleHeight}px`;
    leftOverlay.style.backgroundColor = 'rgba(0, 0, 255, 0)';
    leftOverlay.style.cursor = 'w-resize';
    leftOverlay.style.zIndex = '1000';
    
    // Set right handle position and size
    rightOverlay.style.position = 'absolute';
    rightOverlay.style.left = `${pixelX1 - handleWidth}px`;
    rightOverlay.style.width = `${handleWidth}px`;
    rightOverlay.style.top = `${yAxis._offset}px`;
    rightOverlay.style.height = `${handleHeight}px`;
    rightOverlay.style.backgroundColor = 'rgba(0, 0, 255, 0)';
    rightOverlay.style.cursor = 'e-resize';
    rightOverlay.style.zIndex = '1000';
}

// Helper function to hide overlay that doesn't match current labeling
function hideOverlay(index) {
    const dragOverlay = document.getElementById(`drag-overlay-${index}`);
    const leftOverlay = document.getElementById(`left-overlay-${index}`);
    const rightOverlay = document.getElementById(`right-overlay-${index}`);
    
    if (dragOverlay) dragOverlay.style.display = 'none';
    if (leftOverlay) leftOverlay.style.display = 'none';
    if (rightOverlay) rightOverlay.style.display = 'none';
}

// Toggle splitting mode
function toggleSplitMode() {
    isSplitting = !isSplitting;
    const split_btn_overlay = document.getElementById('split-btn-overlay');
    split_btn_overlay.style.background = isSplitting ? 'rgba(224, 224, 224)' : 'rgba(0, 0, 0, 0)';
}
async function toggleVerifiedStatus() {
    // Get current session
    const session = getCurrentSession();
    if (!session) {
        console.error('No current session to toggle verified status');
        return;
    }
    
    // Get the verified button (try visualization view first, then table view)
    let verified_btn_viz = document.getElementById('verified-btn-viz');
    if (!verified_btn_viz) {
        verified_btn_viz = document.getElementById(`verified-btn-${session.session_id}`);
    }
    
    if (!verified_btn_viz) {
        console.error('Verified button not found');
        return;
    }
    
    // Toggle verified status
    session.verified = session.verified ? 0 : 1;
    
    // Update the visual state immediately
    verified_btn_viz.style.color = session.verified ? '#28a745' : '#dee2e6';
    
    // Save to backend
    try {
        await SessionAPI.updateSessionMetadata(session);
        console.log(`Session ${session.session_id} verified status updated to: ${session.verified}`);
        
        // Also update the sessions list if we're in table view or sidebar
        updateSessionsList();

    } catch (error) {
        console.error('Error updating verified status:', error);
        // Revert the visual change on error
        session.verified = session.verified ? 0 : 1;
        verified_btn_viz.style.color = session.verified ? '#28a745' : '#dee2e6';
    }
}

function getCurrentSession() {
    return SessionService.findSessionById(sessions, currentSessionId);
}

 function deleteAllBouts(currentSessionId) {
        if (dragContext.currentSession && dragContext.currentSession.bouts) {
            // Get current zoom level before removing bout
            const plotDiv = document.getElementById('timeSeriesPlot');
            let viewState = null;
            if (plotDiv && plotDiv._fullLayout && plotDiv._fullLayout.xaxis) {
                viewState = {
                    xrange: plotDiv._fullLayout.xaxis.range.slice(),
                    yrange: plotDiv._fullLayout.yaxis.range.slice()
                };
            }
            
            dragContext.currentSession.bouts.splice(0);
            console.log("Removed all bouts");
            SessionAPI.updateSessionMetadata(dragContext.currentSession);
            
            // Refresh the plot and restore zoom level
            visualizeSession(currentSessionId).then(() => {
                if (viewState && plotDiv) {
                    Plotly.relayout(plotDiv, {
                        'xaxis.range': viewState.xrange,
                        'yaxis.range': viewState.yrange
                    });
                }
            });
        }
}

async function decideSession(sessionId, keep) {
    const session = sessions.find(s => s.session_id == sessionId);
    if (!session) return;
    
    const wasCurrentlyVisualized = (currentSessionId == sessionId);
    
    // Find the next session before marking this one as deleted
    let nextSessionToShow = null;
    if (!keep && wasCurrentlyVisualized) {
        nextSessionToShow = SessionService.findNextSessionAfterDeletion(sessions, sessionId);
    }
    
    // Update the session status
    session.status = "Decision Made";
    // Explicitly set keep to 0 or 1, rather than false/true
    session.keep = keep ? 1 : 0;
    await SessionAPI.updateSessionMetadata(session);
    
    // Update the UI with the new session list
    updateSessionsList();
    
    // Handle navigation after deletion
    if (!keep && wasCurrentlyVisualized) {
        if (nextSessionToShow && nextSessionToShow.session_id != sessionId) {
            // Navigate to the next session in sequence
            console.log(`Deleted currently visualized session, navigating to: ${nextSessionToShow.session_name}`);
            visualizeSession(nextSessionToShow.session_id);
        } else {
            // If we couldn't find a proper next session or only this session exists
            const remainingSessions = SessionService.getFilteredSessions(sessions);
            if (remainingSessions.length > 0) {
                console.log(`Navigating to first available session: ${remainingSessions[0].session_name}`);
                visualizeSession(remainingSessions[0].session_id);
            } else {
                // No sessions left, return to table view
                console.log('No sessions available after deletion, returning to table view');
                showTableView();
            }
        }
    } else {
        // Return to table view for non-current sessions or when keeping sessions
        showTableView();
    }
}

// Now update the splitSession function to maintain context
async function splitSession() {
    console.log('Splitting session:', currentSessionId);
    if (splitPoints.length === 0) {
        alert('No split points selected');
        return;
    }
    
    // First, ensure any pending bout changes are saved
    const currentSession = sessions.find(s => s.session_id == currentSessionId);
    if (currentSession) {
        try {
            // Wait for metadata update to complete before proceeding
            await SessionAPI.updateSessionMetadata(currentSession);
            console.log('Ensured latest bout changes are saved before splitting');
        } catch (error) {
            console.error('Error saving bout changes before split:', error);
        }
    }
    
    try {
        const response = await fetch(`/api/session/${currentSessionId}/split`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ split_points: splitPoints })
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to split session');
        }
        const result = await response.json();
        // Clear split points
        splitPoints = [];
        isSplitting = false;

        // PROBLEM: This only resets colors, not the expanded state
        document.querySelectorAll('#project-dropdown-menu .dropdown-item').forEach(item => {
            if (!item.classList.contains('active')) {
                item.style.backgroundColor = '';
                item.style.color = '';
            }
        });

        // Reset Bootstrap dropdown state
        const dropdownToggle = document.querySelector('[data-bs-toggle="dropdown"]');
        if (dropdownToggle) {
            dropdownToggle.setAttribute('aria-expanded', 'false');
            const dropdownMenu = document.getElementById('project-dropdown-menu');
            if (dropdownMenu) dropdownMenu.classList.remove('show');
            dropdownToggle.classList.remove('show');
        }

        // Fetch sessions based on current context
        if (currentProjectId) {
            await ProjectController.fetchProjectSessions(currentProjectId);
        } else {
            await ProjectController.fetchSessions();
        }
        
        showTableView();
    } catch (error) {
        console.error('Error splitting session:', error);
        alert('Failed to split session: ' + error.message);
    }
}

function createNewBout() {
    // Get current labeling name
    const currentLabelingElement = document.getElementById('current-labeling-name');
    let currentLabelingName = '';
    
    if (currentLabelingElement) {
        // Extract text content, removing any HTML elements
        const textContent = currentLabelingElement.textContent || currentLabelingElement.innerText;
        currentLabelingName = textContent.trim();
    }

    // Check if currentLabelingName is "No Labeling" and do nothing
    if (currentLabelingName === "No Labeling") {
        console.log('Cannot create bout: No labeling selected');
        return;
    }

    // Get the current plotly visualization
    const plotDiv = document.getElementById('timeSeriesPlot');
    
    // If there's no active plot, fall back to the full dataset range
    if (!plotDiv || !plotDiv._fullLayout || !plotDiv._fullLayout.xaxis) {
        // Fallback to using full dataset range with 25% width
        const middleTimestamp = (minTimestamp + maxTimestamp) / 2;
        const fullRange = maxTimestamp - minTimestamp;
        const boutHalfWidth = (fullRange * 0.25) / 2;
        const newBout = [middleTimestamp - boutHalfWidth, middleTimestamp + boutHalfWidth];
        addBoutToSession(newBout);
        return;
    }
    
    // Get the current visible x-axis range
    const xAxis = plotDiv._fullLayout.xaxis;
    const visibleMin = xAxis.range[0];
    const visibleMax = xAxis.range[1];
    
    // Save current zoom level
    const currentViewState = {
        xrange: [visibleMin, visibleMax],
        yrange: plotDiv._fullLayout.yaxis.range.slice()
    };
    
    // Calculate the midpoint of the visible range
    const middleTimestamp = (visibleMin + visibleMax) / 2;
    
    // Create a bout that is 25% of the visible range (12.5% on each side of the midpoint)
    const visibleRange = visibleMax - visibleMin;
    const boutHalfWidth = (visibleRange * 0.25) / 2;

    // Create a 480s wide bout (240s on each side of the midpoint)
    const newBout = {'start':middleTimestamp - boutHalfWidth,'end':middleTimestamp + boutHalfWidth,'label':currentLabelingName};
    
    // Add the bout to the session and maintain view state
    addBoutToSession(newBout, currentViewState);
}

// Helper function to add bout to session and update display
function addBoutToSession(newBout, viewState = null) {
    if (dragContext.currentSession && dragContext.currentSession.bouts) {
        // Add the new bout to the session
        dragContext.currentSession.bouts.push(newBout);
        
        // Update the session metadata in the background
        SessionAPI.updateSessionMetadata(dragContext.currentSession).then(() => {
            // Instead of full reloading, just add the new overlay
            const container = document.querySelector('.plot-container');
            
            // Create and add the new overlay elements
            const boutIndex = dragContext.currentSession.bouts.length - 1;
            createBoutOverlays(boutIndex, container);
            
            // Position the new overlay
            const plotDiv = document.getElementById('timeSeriesPlot');
            updateOverlayPositions(plotDiv, newBout, boutIndex);
            
            // Apply the previous view state if provided
            if (viewState) {
                Plotly.relayout(plotDiv, {
                    'xaxis.range': viewState.xrange,
                    'yaxis.range': viewState.yrange
                });
            }
        }).catch(error => {
            console.error('Error saving new bout:', error);
        });
    } else {
        console.error('No current session in drag context');
    }
}


async function exportLabelsJSON() {
    try {
        const response = await fetch('/api/export/labels');
        if (!response.ok) {
            throw new Error('Failed to export data');
        }
        
        const data = await response.json();
        
        // Create downloadable JSON file
        const jsonString = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        // Create download link
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0] + '_' + 
                         new Date().toISOString().replace(/[:.]/g, '-').split('T')[1].split('-')[0];
        const filename = `smoking_labels_export_${timestamp}.json`;
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        // Success - file downloaded automatically, no need for alert/toast
        console.log(`Successfully exported ${data.total_sessions} sessions with ${data.total_labels} labels to ${filename}`);
        
    } catch (error) {
        console.error('Error exporting JSON:', error);
        // Show error notification only
        showNotification('Failed to export data: ' + error.message, 'error');
    }
}

function updateSidebarHighlighting() {
    // Remove active-session class from all links
    document.querySelectorAll('#session-list .nav-link').forEach(link => {
        link.classList.remove('active-session');
    });
    
    // Add active-session class to the current session
    if (currentActiveSession) {
        const activeLink = document.querySelector(`#session-list .nav-link[onclick*="'${currentSessionId}'"]`);
        if (activeLink) {
            activeLink.classList.add('active-session');
        }
    }
}



// Function to open color picker when circle is clicked
function openColorPicker(labelingName, circleElement) {
    const colorPicker = circleElement.parentElement.querySelector('.color-picker');
    if (colorPicker) {
        colorPicker.click();
    }
}

// Bulk upload form event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Handle form submission
    const bulkUploadForm = document.getElementById('bulk-upload-form');
    if (bulkUploadForm) {
        bulkUploadForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const bulkUploadFolderPath = document.getElementById('bulk-upload-folder').value;
            
            console.log('Starting bulk upload for folder:', bulkUploadFolderPath);
            ProjectController.createBulkUpload(bulkUploadFolderPath);
        });
    }
    
    // Reset modal when it's closed
    const bulkUploadModal = document.getElementById('bulkUploadModal');
    if (bulkUploadModal) {
        bulkUploadModal.addEventListener('hidden.bs.modal', function() {
            // Reset form and hide progress
            const form = document.getElementById('bulk-upload-form');
            const progress = document.getElementById('bulk-upload-progress');
            const preview = document.getElementById('bulk-preview');
            
            if (form) form.reset();
            if (progress) progress.style.display = 'none';
            if (preview) preview.style.display = 'none';
            if (form) form.style.display = 'block';
        });
    }
});

// Make functions available globally for inline event handlers
window.visualizeSession = visualizeSession;
window.openColorPicker = openColorPicker;
window.updateLabelingColor = ProjectController.updateLabelingColor;
window.editLabeling = ProjectController.editLabeling;
window.duplicateLabeling = ProjectController.duplicateLabeling;
window.deleteLabeling = ProjectController.deleteLabeling;
window.selectLabeling = selectLabeling;
window.deleteAllBouts = deleteAllBouts;
window.scoreSession = SessionController.scoreSession;
window.showTableView = showTableView;
window.decideSession = decideSession;
window.toggleSplitMode = toggleSplitMode;
window.toggleVerifiedStatus = toggleVerifiedStatus;
window.splitSession = splitSession;
window.createNewBout = createNewBout;
window.showCreateProjectForm = showCreateProjectForm;
window.createNewProject = ProjectController.createNewProject;
window.createBulkUpload = ProjectController.createBulkUpload;
window.showBulkUploadForm = showBulkUploadForm;
window.updateSidebarHighlighting = updateSidebarHighlighting;
window.updateSessionsList = updateSessionsList;
window.exportLabelsJSON = exportLabelsJSON;
window.showBulkUploadForm = showBulkUploadForm;
window.pollScoringStatus = pollScoringStatus;
window.deleteProject = ProjectController.deleteProject;

// Export overlay management functions for overlay manager
window.updateOverlayPositions = updateOverlayPositions;
window.hideOverlay = hideOverlay;
window.createBoutOverlays = createBoutOverlays;

ProjectController.initializeProjects();
eventListeners.addEventListeners();
checkUrlParameters();
checkUrlParameters();