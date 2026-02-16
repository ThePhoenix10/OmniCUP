function displayTherapyResults(result) {
    const container = document.getElementById('therapyResults');
    const resultsContainer = document.getElementById('therapyResultsContainer');
    
    container.innerHTML = '';
    
    console.log('Full result from backend:', result);
    
    let outputText = result.clinical_analysis || result.raw_query_response || 'No analysis available.';
    
    const resultDiv = document.createElement('div');
    resultDiv.className = 'therapy-result-text';
    resultDiv.style.cssText = `
        background: #f9fafb;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 20px;
        font-size: 13px;
        line-height: 1.7;
        color: #111827;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
        max-height: 900px;
        overflow-y: auto;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    `;
    resultDiv.textContent = outputText;
    
    container.appendChild(resultDiv);
    
    resultsContainer.style.display = 'block';
    
    setTimeout(() => {
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

let GENE_OPTIONS = [];
let HOTSPOT_FEATURES = [];
let ENSEMBLE_WEIGHTS = {};

const DMETS_UI_TO_MODEL = {
    "Distant Lymph Nodes": "DMETS_DX_DIST_LN",
    "Lungs": "DMETS_DX_LUNG",
    "Pleura": "DMETS_DX_PLEURA",
    "Mediastinum": "DMETS_DX_MEDIASTINUM",
    "Liver": "DMETS_DX_LIVER",
    "Biliary Tract": "DMETS_DX_BILIARY_TRACT",
    "Abdominal Cavity": "DMETS_DX_INTRA_ABDOMINAL",
    "Bowel": "DMETS_DX_BOWEL",
    "Brain / CNS": "DMETS_DX_CNS_BRAIN",
    "Bones": "DMETS_DX_BONE",
    "Peripheral Nervous System": "DMETS_DX_PNS",
    "Adrenal Glands": "DMETS_DX_ADRENAL_GLAND",
    "Kidneys": "DMETS_DX_KIDNEY",
    "Bladder / Urinary Tract": "DMETS_DX_BLADDER_UT",
    "Female Reproductive Organs": "DMETS_DX_FEMALE_GENITAL",
    "Ovaries": "DMETS_DX_OVARY",
    "Male Reproductive Organs": "DMETS_DX_MALE_GENITAL",
    "Skin": "DMETS_DX_SKIN",
    "Head and Neck": "DMETS_DX_HEAD_NECK",
    "Breast Tissue": "DMETS_DX_BREAST",
    "Unspecified Site": "DMETS_DX_UNSPECIFIED"
};

const DMETS_SITES = Object.keys(DMETS_UI_TO_MODEL);

const state = {
    genes: [],
    hotspots: new Set(),
    dmets: {}
};

DMETS_SITES.forEach(site => {
    state.dmets[site] = false;
});

async function loadData() {
    try {
        const response = await fetch('/api/data');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        GENE_OPTIONS = data.genes;
        HOTSPOT_FEATURES = data.hotspots;
        ENSEMBLE_WEIGHTS = data.ensemble_weights;
        
        setupAutocomplete('geneInput', 'geneList', GENE_OPTIONS, addGene);
        setupAutocomplete('hotspotInput', 'hotspotList', HOTSPOT_FEATURES, addHotspot);
        
        renderDmets();
    } catch (error) {
        console.error('Failed to load data:', error);
        showStatus('Error loading data from server: ' + error.message, 'error');
    }
}

function switchTab(tabName, e) {
    e.preventDefault();
    
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    const buttons = document.querySelectorAll('.tab-button');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabName + '-tab').classList.add('active');
    
    e.currentTarget.classList.add('active');
}

function showStatus(message, type) {
    const el = document.getElementById('statusMessage');
    el.textContent = message;
    el.className = `status-message ${type}`;
    setTimeout(() => {
        el.className = 'status-message';
    }, 4000);
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('show', show);
}

function showTherapyLoading(show) {
    const container = document.getElementById('therapyLoadingContainer');
    container.style.display = show ? 'block' : 'none';
}

function showTherapyError(message) {
    const errorContainer = document.getElementById('therapyErrorContainer');
    const errorMessage = document.getElementById('therapyErrorMessage');
    errorMessage.textContent = message;
    errorContainer.style.display = 'block';
    
    setTimeout(() => {
        errorContainer.style.display = 'none';
    }, 5000);
}

function setupAutocomplete(inputId, listId, options, callback) {
    const input = document.getElementById(inputId);
    const list = document.getElementById(listId);

    input.addEventListener('input', function() {
        const value = this.value.trim().toUpperCase();
        list.innerHTML = '';

        if (!value) {
            list.style.display = 'none';
            return;
        }

        const filtered = options.filter(o => o.toUpperCase().includes(value)).slice(0, 15);

        filtered.forEach(option => {
            const item = document.createElement('div');
            item.className = 'autocomplete-item';
            item.textContent = option;
            item.addEventListener('click', function() {
                callback(option);
                input.value = '';
                list.style.display = 'none';
            });
            list.appendChild(item);
        });

        list.style.display = filtered.length > 0 ? 'block' : 'none';
    });

    document.addEventListener('click', function(e) {
        if (e.target !== input && e.target !== list) {
            list.style.display = 'none';
        }
    });
}

function addGene(gene) {
    state.genes.push({
        name: gene,
        LOF: false,
        NON_LOF: false,
        AMP: false,
        DEL: false,
        FUSION: false
    });

    renderGeneTable();
    updateSummary();
}

function removeGene(index) {
    state.genes.splice(index, 1);
    renderGeneTable();
    updateSummary();
}

function renderGeneTable() {
    const tbody = document.getElementById('geneTableBody');

    if (state.genes.length === 0) {
        tbody.innerHTML = '<tr class="empty-state"><td colspan="7">No genes added yet. Search and select a gene above.</td></tr>';
        return;
    }

    tbody.innerHTML = '';

    state.genes.forEach((geneData, index) => {
        const row = document.createElement('tr');
        const gene = geneData.name;

        row.innerHTML = `
            <td><strong>${gene}</strong></td>
            <td class="checkbox-cell">
                <input type="checkbox" data-index="${index}" data-type="LOF" 
                       ${geneData.LOF ? 'checked' : ''} onchange="updateGeneAlteration(${index}, 'LOF', this.checked)">
            </td>
            <td class="checkbox-cell">
                <input type="checkbox" data-index="${index}" data-type="NON_LOF" 
                       ${geneData.NON_LOF ? 'checked' : ''} onchange="updateGeneAlteration(${index}, 'NON_LOF', this.checked)">
            </td>
            <td class="checkbox-cell">
                <input type="checkbox" data-index="${index}" data-type="AMP" 
                       ${geneData.AMP ? 'checked' : ''} onchange="updateGeneAlteration(${index}, 'AMP', this.checked)">
            </td>
            <td class="checkbox-cell">
                <input type="checkbox" data-index="${index}" data-type="DEL" 
                       ${geneData.DEL ? 'checked' : ''} onchange="updateGeneAlteration(${index}, 'DEL', this.checked)">
            </td>
            <td class="checkbox-cell">
                <input type="checkbox" data-index="${index}" data-type="FUSION" 
                       ${geneData.FUSION ? 'checked' : ''} onchange="updateGeneAlteration(${index}, 'FUSION', this.checked)">
            </td>
            <td style="text-align: center;">
                <button class="btn btn-remove" onclick="removeGene(${index})">Remove</button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function updateGeneAlteration(index, type, checked) {
    const geneData = state.genes[index];
    
    if (checked) {
        Object.keys(geneData).forEach(key => {
            if (key !== 'name') {
                geneData[key] = false;
            }
        });
        geneData[type] = true;
    } else {
        geneData[type] = false;
    }

    renderGeneTable();
    updateSummary();
}

function renderDmets() {
    const container = document.getElementById('dmetsGrid');
    container.innerHTML = '';

    DMETS_SITES.forEach(site => {
        const item = document.createElement('div');
        item.className = 'dmets-item';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `dmets_${site}`;
        checkbox.checked = state.dmets[site] || false;
        checkbox.addEventListener('change', function() {
            state.dmets[site] = this.checked;
            updateSummary();
        });

        const label = document.createElement('label');
        label.htmlFor = `dmets_${site}`;
        label.className = 'dmets-label';
        label.textContent = site;

        item.appendChild(checkbox);
        item.appendChild(label);
        container.appendChild(item);
    });
}

function addHotspot(hotspot) {
    if (state.hotspots.has(hotspot)) {
        return;
    }

    state.hotspots.add(hotspot);
    renderHotspots();
    updateSummary();
}

function removeHotspot(hotspot) {
    state.hotspots.delete(hotspot);
    renderHotspots();
    updateSummary();
}

function renderHotspots() {
    const container = document.getElementById('hotspotBadges');
    const hotspots = Array.from(state.hotspots).sort();

    if (hotspots.length === 0) {
        container.innerHTML = '<span class="empty-state">No hotspots selected yet.</span>';
        return;
    }

    container.innerHTML = '';

    hotspots.forEach(hotspot => {
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.innerHTML = `
            ${hotspot}
            <span class="badge-close" onclick="removeHotspot('${hotspot}')">✕</span>
        `;
        container.appendChild(badge);
    });
}

function updateSummary() {
    const modelDmets = {};
    
    Object.entries(state.dmets).forEach(([uiName, value]) => {
        if (value) {
            const modelName = DMETS_UI_TO_MODEL[uiName];
            if (modelName) {
                modelDmets[modelName] = 1;
            }
        }
    });

    const genesObj = {};
    state.genes.forEach(geneData => {
        if (!genesObj[geneData.name]) {
            genesObj[geneData.name] = {};
        }
        genesObj[geneData.name] = {
            LOF: geneData.LOF,
            NON_LOF: geneData.NON_LOF,
            AMP: geneData.AMP,
            DEL: geneData.DEL,
            FUSION: geneData.FUSION
        };
    });

    const data = {
        Clinical: {
            Age: document.getElementById('age').value || null,
            Sex: document.getElementById('sex').value || null,
            TMB: document.getElementById('tmb').value || null,
            MSI: document.getElementById('msi').value || null
        },
        Genes: genesObj,
        Hotspots: Array.from(state.hotspots).sort(),
        DMETS: modelDmets
    };

    console.log('Summary data:', data);
}

function submitData() {
    if (state.genes.length === 0) {
        showStatus('Please add at least one gene before submitting.', 'error');
        return;
    }

    const modelDmets = {};
    
    Object.entries(state.dmets).forEach(([uiName, value]) => {
        if (value) {
            const modelName = DMETS_UI_TO_MODEL[uiName];
            if (modelName) {
                modelDmets[modelName] = 1;
            }
        }
    });

    const genesObj = {};
    state.genes.forEach(geneData => {
        if (!genesObj[geneData.name]) {
            genesObj[geneData.name] = {};
        }
        genesObj[geneData.name] = {
            LOF: geneData.LOF,
            NON_LOF: geneData.NON_LOF,
            AMP: geneData.AMP,
            DEL: geneData.DEL,
            FUSION: geneData.FUSION
        };
    });

    // Convert empty clinical fields to null (backend converts to NaN)
    const getClinicalValue = (inputId) => {
        const value = document.getElementById(inputId).value;
        return (value !== null && value !== "") ? value : null;
    };

    const data = {
        Clinical: {
            Age: getClinicalValue('age'),
            Sex: document.getElementById('sex').value || null,
            TMB: getClinicalValue('tmb'),
            MSI: getClinicalValue('msi')
        },
        Genes: genesObj,
        Hotspots: Array.from(state.hotspots).sort(),
        DMETS: modelDmets
    };

    showLoading(true);

    fetch('/api/debug-patient', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    }).catch(err => console.error('Debug logging error:', err));

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    })
    .then(result => {
        showLoading(false);
        
        if (result.status === 'success') {
            displayPredictions(result.predictions);
        } else {
            showStatus(`Error: ${result.message}`, 'error');
        }
        
        console.log('Server response:', result);
    })
    .catch(error => {
        showLoading(false);
        showStatus(`Error: ${error.message}`, 'error');
        console.error('Error:', error);
    });
}

function displayPredictions(predictions) {
    const container = document.getElementById('predictionsList');
    const resultsContainer = document.getElementById('resultsContainer');
    
    container.innerHTML = '';
    
    predictions.forEach(pred => {
        const card = document.createElement('div');
        card.className = 'prediction-card';
        card.innerHTML = `
            <div class="prediction-rank">${pred.rank}</div>
            <div class="prediction-info">
                <div class="prediction-cancer-type">${pred.cancer_type}</div>
                <div class="prediction-confidence">Confidence: ${pred.confidence}</div>
            </div>
            <div class="prediction-probability">${pred.confidence}</div>
        `;
        container.appendChild(card);
    });
    
    resultsContainer.style.display = 'block';
}

function submitTherapyLookup() {
    const gene = document.getElementById('therapyGene').value.trim();
    const cancerType = document.getElementById('therapyCancerType').value.trim();
    
    document.getElementById('therapyResultsContainer').style.display = 'none';
    document.getElementById('therapyErrorContainer').style.display = 'none';
    
    if (!gene || !cancerType) {
        showTherapyError('Please enter both gene and cancer type');
        return;
    }
    
    console.log('Submitting therapy lookup:', { gene, cancerType });
    
    showTherapyLoading(true);
    
    fetch('/api/therapy-lookup', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            gene: gene,
            cancer_type: cancerType
        })
    })
    .then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    })
    .then(result => {
        showTherapyLoading(false);
        console.log('Therapy lookup result:', result);
        
        if (result.status === 'success') {
            displayTherapyResults(result);
        } else {
            showTherapyError(`Error: ${result.message}`);
        }
    })
    .catch(error => {
        showTherapyLoading(false);
        showTherapyError(`Error: ${error.message}`);
        console.error('Therapy lookup error:', error);
    });
}

document.addEventListener('DOMContentLoaded', function() {
    loadData();

    ['age', 'sex', 'tmb', 'msi'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', updateSummary);
        }
    });

    updateSummary();
});

document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.ctrlKey) {
        submitData();
    }
});