// Frontend JavaScript for EEG Integration
// This shows how to use the new async Celery endpoints

class EEGSessionManager {
    constructor() {
        this.currentTaskId = null;
        this.statusCheckInterval = null;
    }

    // Start EEG session with specified duration
    async startSession(durationMinutes = 1) {
        try {
            const response = await fetch('/start_eeg/', {
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
                this.startStatusPolling();
                console.log('EEG session started:', result);
                return result;
            } else {
                throw new Error(result.error || 'Failed to start session');
            }
        } catch (error) {
            console.error('Error starting EEG session:', error);
            throw error;
        }
    }

    // Start polling for task status
    startStatusPolling() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
        }

        this.statusCheckInterval = setInterval(async () => {
            try {
                const status = await this.checkStatus();
                this.updateUI(status);
                
                if (status.status === 'completed' || status.status === 'error') {
                    this.stopStatusPolling();
                }
            } catch (error) {
                console.error('Error checking status:', error);
            }
        }, 2000); // Check every 2 seconds
    }

    // Stop polling
    stopStatusPolling() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
        }
    }

    // Check current task status
    async checkStatus() {
        try {
            const response = await fetch('/eeg_status/', {
                method: 'GET',
                headers: {
                    'X-CSRFToken': this.getCSRFToken()
                }
            });

            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Error checking status:', error);
            throw error;
        }
    }

    // Get final results
    async getResults() {
        try {
            const response = await fetch('/stop_eeg/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                }
            });

            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Error getting results:', error);
            throw error;
        }
    }

    // Update UI based on status
    updateUI(status) {
        const statusElement = document.getElementById('eeg-status');
        const progressElement = document.getElementById('eeg-progress');
        const resultsElement = document.getElementById('eeg-results');

        if (statusElement) {
            switch (status.status) {
                case 'processing':
                case 'PENDING':
                    statusElement.textContent = 'EEG recording in progress...';
                    statusElement.className = 'status-processing';
                    break;
                case 'completed':
                case 'SUCCESS':
                    statusElement.textContent = 'EEG recording completed!';
                    statusElement.className = 'status-completed';
                    this.displayResults(status.result);
                    break;
                case 'error':
                case 'FAILURE':
                    statusElement.textContent = 'Error: ' + (status.error || 'Unknown error');
                    statusElement.className = 'status-error';
                    break;
            }
        }
    }

    // Display final results
    displayResults(result) {
        const resultsElement = document.getElementById('eeg-results');
        if (resultsElement && result) {
            resultsElement.innerHTML = `
                <h3>Session Results</h3>
                <p><strong>Session ID:</strong> ${result.session_id}</p>
                <p><strong>Predicted State:</strong> ${result.final_result?.predicted_label || 'N/A'}</p>
                <p><strong>Confidence:</strong> ${(result.final_result?.confidence || 0).toFixed(2)}</p>
                <p><strong>Total Duration:</strong> ${result.final_result?.total_seconds || 0} seconds</p>
                <p><strong>Relaxed:</strong> ${result.final_result?.relaxed_seconds || 0} seconds</p>
                <p><strong>Neutral:</strong> ${result.final_result?.neutral_seconds || 0} seconds</p>
                <p><strong>Concentrating:</strong> ${result.final_result?.concentrating_seconds || 0} seconds</p>
            `;
        }
    }

    // Get CSRF token from cookies
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
}

// Example usage:
const eegManager = new EEGSessionManager();

// Start a 2-minute session when button is clicked
document.getElementById('start-eeg-btn')?.addEventListener('click', async () => {
    try {
        await eegManager.startSession(2);
        document.getElementById('start-eeg-btn').disabled = true;
    } catch (error) {
        alert('Failed to start EEG session: ' + error.message);
    }
});

// Get results when button is clicked
document.getElementById('get-results-btn')?.addEventListener('click', async () => {
    try {
        const results = await eegManager.getResults();
        console.log('Final results:', results);
    } catch (error) {
        alert('Failed to get results: ' + error.message);
    }
});
