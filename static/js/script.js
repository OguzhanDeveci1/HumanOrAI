// DOM Elements
const textInput = document.getElementById('textInput');
const charCount = document.getElementById('charCount');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');

// Ensemble elements
const ensembleIcon = document.getElementById('ensembleIcon');
const ensembleLabel = document.getElementById('ensembleLabel');
const ensembleConfidence = document.getElementById('ensembleConfidence');
const ensembleVotes = document.getElementById('ensembleVotes');
const ensembleBar = document.getElementById('ensembleBar');

// Character count update
textInput.addEventListener('input', () => {
    const count = textInput.value.length;
    charCount.textContent = count.toLocaleString();
});

// Clear button
clearBtn.addEventListener('click', () => {
    textInput.value = '';
    charCount.textContent = '0';
    hideResults();
    hideError();
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();

    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }

    // Validate minimum word count (50 words required)
    const wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
    if (wordCount < 50) {
        showError(`Text must contain at least 50 words. Current: ${wordCount} words.`);
        return;
    }

    hideError();
    hideResults();
    showLoading();

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to analyze text');
        }

        hideLoading();
        displayResults(data);
    } catch (error) {
        hideLoading();
        showError(error.message || 'An error occurred while analyzing the text');
    }
});

// Helper functions
function showLoading() {
    loadingState.style.display = 'block';
    analyzeBtn.disabled = true;
}

function hideLoading() {
    loadingState.style.display = 'none';
    analyzeBtn.disabled = false;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => hideError(), 5000);
}

function hideError() {
    errorMessage.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function displayResults(data) {
    // Store latest results for statistics
    latestResults = data;

    const { individual_results, ensemble } = data;

    // Display ensemble result
    displayEnsembleResult(ensemble);

    // Display individual model results
    displayIndividualResults(individual_results);

    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayEnsembleResult(ensemble) {
    const isHuman = ensemble.label === 'HUMAN';

    // Set icon
    ensembleIcon.innerHTML = isHuman
        ? '👤' // Human icon
        : '🤖'; // Robot icon

    // Set label
    ensembleLabel.textContent = ensemble.label;
    ensembleLabel.className = `ensemble-label ${isHuman ? 'human' : 'ai'}`;

    // Set confidence
    ensembleConfidence.textContent = `${ensemble.confidence}%`;

    // Set votes
    ensembleVotes.textContent = `${ensemble.vote_count}/${ensemble.total_models}`;

    // Set confidence bar
    ensembleBar.style.width = `${ensemble.confidence}%`;
}

function displayIndividualResults(results) {
    const models = ['BERT', 'RoBERTa', 'DRF', 'GBM', 'GLM'];

    models.forEach(modelName => {
        const result = results[modelName];
        const modelCard = document.getElementById(`model-${modelName}`);

        if (!modelCard || !result) return;

        const isHuman = result.label === 'HUMAN';

        // Update prediction badge
        const badge = modelCard.querySelector('.prediction-badge');
        badge.textContent = result.label;
        badge.className = `prediction-badge ${isHuman ? 'human' : 'ai'}`;

        // Update confidence value
        const confidenceValue = modelCard.querySelector('.confidence-value');
        confidenceValue.textContent = `${result.confidence}%`;

        // Update progress bar
        const progressFill = modelCard.querySelector('.progress-fill');
        progressFill.style.width = `${result.confidence}%`;
        progressFill.className = `progress-fill ${isHuman ? 'human' : 'ai'}`;
    });
}

// Sample texts for testing (optional feature)
const sampleTexts = {
    human: `The art of writing has been a fundamental part of human civilization for thousands of years. From ancient cave paintings to modern digital communication, our desire to express thoughts and share stories has driven innovation and cultural development. Writing allows us to preserve knowledge, share experiences, and connect with others across time and space. It is through writing that we have documented our history, shared scientific discoveries, and created works of literature that inspire and educate generations.`,

    ai: `Artificial intelligence has revolutionized numerous industries by providing efficient solutions to complex problems. Machine learning algorithms can process vast amounts of data and identify patterns that would be impossible for humans to detect manually. This technology has applications in healthcare, finance, transportation, and many other sectors. The continued development of AI systems promises to bring even more significant changes to how we work and live in the future.`
};

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        analyzeBtn.click();
    }

    // Ctrl/Cmd + K to clear
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        clearBtn.click();
    }
});

// Auto-resize textarea
textInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});

// ============ Statistics Dashboard ============

// Store latest prediction results globally
let latestResults = null;
let confidenceChart = null;

// Statistics modal elements
const statsBtn = document.getElementById('statsBtn');
const statsModal = document.getElementById('statsModal');
const closeModal = document.getElementById('closeModal');
const modelRankings = document.getElementById('modelRankings');

// Open statistics modal
statsBtn.addEventListener('click', () => {
    if (!latestResults) {
        showError('No predictions yet. Analyze some text first!');
        return;
    }

    statsModal.style.display = 'flex';
    renderPieChart();
    renderRankings();
});

// Close modal - X button
closeModal.addEventListener('click', () => {
    statsModal.style.display = 'none';
});

// Close modal - backdrop click
statsModal.addEventListener('click', (e) => {
    if (e.target === statsModal) {
        statsModal.style.display = 'none';
    }
});

// Close modal - Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && statsModal.style.display === 'flex') {
        statsModal.style.display = 'none';
    }
});

// Render pie chart with Chart.js
function renderPieChart() {
    const ctx = document.getElementById('confidenceChart').getContext('2d');

    // Extract model data
    const models = ['BERT', 'RoBERTa', 'DRF', 'GBM', 'GLM'];
    const modelData = models.map(name => ({
        name: name,
        confidence: latestResults.individual_results[name].confidence
    }));

    // Destroy existing chart if exists
    if (confidenceChart) {
        confidenceChart.destroy();
    }

    // Create new chart
    confidenceChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: modelData.map(m => m.name),
            datasets: [{
                data: modelData.map(m => m.confidence),
                backgroundColor: [
                    'rgba(102, 126, 234, 0.8)',  // BERT - Primary blue
                    'rgba(118, 75, 162, 0.8)',   // RoBERTa - Purple
                    'rgba(237, 100, 166, 0.8)',  // DRF - Pink
                    'rgba(255, 159, 67, 0.8)',   // GBM - Orange
                    'rgba(72, 219, 251, 0.8)'    // GLM - Cyan
                ],
                borderColor: [
                    'rgba(102, 126, 234, 1)',
                    'rgba(118, 75, 162, 1)',
                    'rgba(237, 100, 166, 1)',
                    'rgba(255, 159, 67, 1)',
                    'rgba(72, 219, 251, 1)'
                ],
                borderWidth: 2,
                hoverOffset: 15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#e2e8f0',
                        font: {
                            size: 13,
                            weight: '500'
                        },
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#e2e8f0',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(102, 126, 234, 0.5)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            return `${label}: ${value.toFixed(2)}%`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// Render model rankings
function renderRankings() {
    const models = ['BERT', 'RoBERTa', 'DRF', 'GBM', 'GLM'];

    // Create array with model data
    const modelData = models.map(name => ({
        name: name,
        confidence: latestResults.individual_results[name].confidence,
        label: latestResults.individual_results[name].label,
        prediction: latestResults.individual_results[name].prediction
    }));

    // Sort by confidence descending
    modelData.sort((a, b) => b.confidence - a.confidence);

    // Generate HTML for rankings
    const rankingsHTML = modelData.map((model, index) => {
        const rank = index + 1;
        const badgeClass = rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'bronze' : 'default';
        const isHuman = model.label === 'HUMAN';
        const labelClass = isHuman ? 'human' : 'ai';
        const icon = isHuman ? '👤' : '🤖';

        // Medal emoji for top 3
        const medal = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : '';

        return `
            <div class="ranking-item" style="animation-delay: ${index * 0.1}s">
                <div class="rank-badge ${badgeClass}">
                    <span class="rank-number">#${rank}</span>
                </div>
                <div class="ranking-info">
                    <div class="ranking-header">
                        <span class="model-name">${model.name}</span>
                        ${medal ? `<span class="medal">${medal}</span>` : ''}
                    </div>
                    <div class="ranking-details">
                        <span class="ranking-prediction ${labelClass}">
                            ${icon} ${model.label}
                        </span>
                        <span class="ranking-confidence">${model.confidence.toFixed(2)}%</span>
                    </div>
                </div>
                <div class="ranking-bar-container">
                    <div class="ranking-bar ${labelClass}" style="width: ${model.confidence}%"></div>
                </div>
            </div>
        `;
    }).join('');

    modelRankings.innerHTML = rankingsHTML;
}
