// Real-time EEG Session Manager with Per-Second Focus Streaming
class RealtimeEEGManager {
    constructor() {
        this.currentTaskId = null;
        this.sessionId = null;
        this.isRealtimeActive = false;
        this.statusCheckInterval = null;
        this.focusData = [];
        this.startTime = null;
        this.firstPredictionTime = null;
        this.timerStarted = false;
    }

    // Start real-time EEG session
    async startRealtimeSession(durationMinutes = 1) {
        try {
            const response = await fetch('/start_realtime_eeg/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify({
                    duration: durationMinutes
                })
            });

            const result = await response.json();
            
            if (result.ok) {
                this.currentTaskId = result.task_id;
                this.sessionId = result.session_id;
                this.isRealtimeActive = true;
                // Don't set startTime yet - wait for first prediction
                this.startTime = null;
                this.firstPredictionTime = null;
                this.timerStarted = false;
                
                // Start real-time status monitoring
                this.startRealtimeMonitoring();
                
                console.log('Real-time EEG session started:', result);
                this.updateUI('initializing', result);
                return result;
            } else {
                throw new Error(result.error || 'Failed to start real-time session');
            }
        } catch (error) {
            console.error('Error starting real-time EEG session:', error);
            throw error;
        }
    }

    // Start monitoring real-time status
    startRealtimeMonitoring() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
        }

        this.statusCheckInterval = setInterval(async () => {
            try {
                const response = await fetch('/realtime_eeg_status/', {
                    method: 'GET',
                    headers: {
                        'X-CSRFToken': this.getCSRFToken()
                    }
                });

                const status = await response.json();
                
                if (status.ok) {
                    this.updateRealtimeUI(status);
                }
            } catch (error) {
                console.error('Error checking real-time status:', error);
            }
        }, 1000); // Check every second for real-time updates
    }

    // Update UI with real-time data
    updateRealtimeUI(status) {
        const statusElement = document.getElementById('realtime-status');
        const focusElement = document.getElementById('current-focus');
        const confidenceElement = document.getElementById('current-confidence');
        const focusScoreElement = document.getElementById('current-focus-score');
        const timerElement = document.getElementById('session-timer');
        const progressBar = document.getElementById('focus-progress');

        if (statusElement) {
            switch (status.status) {
                case 'PENDING':
                case 'initializing':
                    if (!this.timerStarted) {
                        statusElement.textContent = 'Initializing EEG connection... Waiting for first prediction...';
                    } else {
                        statusElement.textContent = 'Session Active - Recording Focus';
                    }
                    statusElement.className = 'status-initializing';
                    break;
                case 'running':
                    statusElement.textContent = 'Session Active - Recording Focus';
                    statusElement.className = 'status-recording';
                    break;
                case 'SUCCESS':
                    statusElement.textContent = 'Session Completed';
                    statusElement.className = 'status-completed';
                    this.isRealtimeActive = false;
                    this.stopRealtimeMonitoring();
                    break;
                case 'FAILURE':
                    statusElement.textContent = 'Session Failed';
                    statusElement.className = 'status-error';
                    this.isRealtimeActive = false;
                    this.stopRealtimeMonitoring();
                    break;
            }
        }

        // Update timer - start only after first prediction
        if (timerElement) {
            if (status.result && status.result.final_summary && !this.timerStarted) {
                // First prediction received - start the timer
                this.firstPredictionTime = new Date();
                this.startTime = this.firstPredictionTime;
                this.timerStarted = true;
                console.log('Timer started at first prediction:', this.startTime);
            }
            
            if (this.startTime) {
                const elapsed = Math.floor((new Date() - this.startTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                timerElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            } else {
                // Show waiting message before first prediction
                timerElement.textContent = '00:00';
            }
        }

        // Update real-time focus data (would come from WebSocket or server-sent events)
        // For now, we'll show the last known state
        if (focusElement && status.result) {
            const summary = status.result.final_summary;
            if (summary) {
                focusElement.textContent = this.getFocusStateText(summary.average_focus_score);
                confidenceElement.textContent = (summary.average_focus_score * 100).toFixed(1) + '%';
                focusScoreElement.textContent = summary.average_focus_score.toFixed(2);
                
                // Update progress bar
                if (progressBar) {
                    const focusPercent = summary.average_focus_score * 100;
                    progressBar.style.width = focusPercent + '%';
                    progressBar.className = this.getProgressBarClass(focusPercent);
                }
            }
        }
    }

    // Stop real-time session
    async stopRealtimeSession() {
        try {
            // Send stop signal to live_predict.py
            const response = await fetch('/end_session/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                }
            });

            const result = await response.json();
            
            if (result.success) {
                // Stop monitoring
                this.isRealtimeActive = false;
                this.stopRealtimeMonitoring();
                
                // Wait a moment for the ML script to save data
                setTimeout(() => {
                    // Redirect to session summary or dashboard
                    window.location.href = '/dashboard/';
                }, 2000);
                
                return result;
            } else {
                throw new Error(result.error || 'Failed to end session');
            }
        } catch (error) {
            console.error('Error ending EEG session:', error);
            throw error;
        }
    }

    // Show final session results
    showFinalResults(result) {
        const resultsContainer = document.getElementById('session-results');
        if (resultsContainer && result.final_summary) {
            const summary = result.final_summary;
            
            resultsContainer.innerHTML = `
                                    <span>${Math.floor(summary.relaxed_seconds)}s</span>
                                </div>
                                <div class="time-item neutral">
                                    <span>Neutral:</span>
                                    <span>${Math.floor(summary.neutral_seconds)}s</span>
                                </div>
                                <div class="time-item concentrating">
                                    <span>Concentrating:</span>
                                    <span>${Math.floor(summary.concentrating_seconds)}s</span>
                                </div>
                            </div>
                        </div>
                        <div class="summary-item">
                            <label>Data Points:</label>
                            <span>${summary.data_points_count}</span>
                        </div>
                        <div class="summary-item">
                            <label>CSV File:</label>
                            <span>${result.csv_file_path}</span>
                        </div>
                    </div>
                </div>
            `;
            
            resultsContainer.style.display = 'block';
        }
    }

    // Get focus state text from score
    getFocusStateText(score) {
        if (score <= 0.4) return 'Relaxed';
        if (score <= 0.7) return 'Neutral';
        return 'Concentrating';
    }

    // Get progress bar class based on focus level
    getProgressBarClass(focusPercent) {
        if (focusPercent <= 40) return 'progress-low';
        if (focusPercent <= 70) return 'progress-medium';
        return 'progress-high';
    }

    // Stop monitoring
    stopRealtimeMonitoring() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
        }
    }

    // Get CSRF token
    getCSRFToken() {
        const cookies = document.cookie.split(';');
        for (let cookie of cookies) {
            const [name, value] = cookie.trim().split('=');
            if (name === 'csrftoken') {
                return decodeURIComponent(value);
            }
        }
        return '';
    }

    // Update UI for general status
    updateUI(status, data) {
        const statusElement = document.getElementById('realtime-status');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `status-${status}`;
        }
    }
}

// Initialize real-time manager
const realtimeEEG = new RealtimeEEGManager();

// Event listeners for real-time controls
document.addEventListener('DOMContentLoaded', () => {
    // Start session button
    document.getElementById('start-realtime-btn')?.addEventListener('click', async () => {
        try {
            const duration = parseInt(document.getElementById('session-duration').value) || 1;
            await realtimeEEG.startRealtimeSession(duration);
            
            // Update UI
            document.getElementById('start-realtime-btn').disabled = true;
            document.getElementById('stop-realtime-btn').disabled = false;
            document.getElementById('session-duration').disabled = true;
        } catch (error) {
            alert('Failed to start EEG session: ' + error.message);
        }
    });

    // Stop session button
    document.getElementById('stop-realtime-btn')?.addEventListener('click', async () => {
        try {
            await realtimeEEG.stopRealtimeSession();
            
            // Update UI
            document.getElementById('start-realtime-btn').disabled = false;
            document.getElementById('stop-realtime-btn').disabled = true;
            document.getElementById('session-duration').disabled = false;
        } catch (error) {
            alert('Failed to stop EEG session: ' + error.message);
        }
    });

    // Initialize button states
    document.getElementById('stop-realtime-btn').disabled = true;
});
