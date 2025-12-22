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
