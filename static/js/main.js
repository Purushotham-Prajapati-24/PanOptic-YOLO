/* ==========================================================================
   🔍 PanOptic-YOLO Interactive Interface Controller (Vanilla JS)
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const progressBarContainer = document.getElementById('progressBarContainer');
    const progressBar = document.getElementById('progressBar');
    
    // Output Cards
    const vizCard = document.getElementById('vizCard');
    const statusCard = document.getElementById('statusCard');
    const ocrCard = document.getElementById('ocrCard');
    const metricsCard = document.getElementById('metricsCard');
    const payloadCard = document.getElementById('payloadCard');
    
    // Outputs elements
    const imgOriginal = document.getElementById('imgOriginal');
    const imgBgRemoved = document.getElementById('imgBgRemoved');
    const imgDetected = document.getElementById('imgDetected');
    
    const statusIndicator = document.getElementById('statusIndicator');
    const statusIcon = document.getElementById('statusIcon');
    const statusTitle = document.getElementById('statusTitle');
    const statusDescription = document.getElementById('statusDescription');
    
    const fieldPan = document.getElementById('fieldPan');
    const fieldName = document.getElementById('fieldName');
    const fieldFather = document.getElementById('fieldFather');
    const fieldDob = document.getElementById('fieldDob');
    
    const valCosine = document.getElementById('valCosine');
    const fillCosine = document.getElementById('fillCosine');
    const valSsim = document.getElementById('valSsim');
    const fillSsim = document.getElementById('fillSsim');
    const valBrightness = document.getElementById('valBrightness');
    const badgeRow = document.getElementById('badgeRow');
    
    const jsonCode = document.getElementById('jsonCode');
    const btnCopyJson = document.getElementById('btnCopyJson');

    /* ==========================================
       1. Drag-and-Drop Handlers
       ========================================== */
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFileUpload(files[0]);
        }
    });

    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });

    /* ==========================================
       2. File Upload & Processing API Call
       ========================================== */
    function handleFileUpload(file) {
        // Reset old views
        progressBarContainer.style.display = 'block';
        progressBar.style.width = '10%';
        
        const formData = new FormData();
        formData.append('file', file);

        // Animate initial upload loading states
        let progressInterval = setInterval(() => {
            let width = parseInt(progressBar.style.width);
            if (width < 85) {
                progressBar.style.width = (width + 5) + '%';
            }
        }, 150);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            clearInterval(progressInterval);
            progressBar.style.width = '100%';
            if (!response.ok) {
                throw new Error('Verification pipeline returned an error.');
            }
            return response.json();
        })
        .then(data => {
            setTimeout(() => {
                progressBarContainer.style.display = 'none';
                updateDashboard(data);
            }, 300);
        })
        .catch(err => {
            clearInterval(progressInterval);
            progressBarContainer.style.display = 'none';
            alert('Error running the detection pipeline: ' + err.message);
        });
    }

    /* ==========================================
       3. Dynamic Dashboard Rendering
       ========================================== */
    function updateDashboard(data) {
        // A. Reveal the dashboard elements
        vizCard.classList.remove('hidden');
        statusCard.classList.remove('hidden');
        ocrCard.classList.remove('hidden');
        metricsCard.classList.remove('hidden');
        payloadCard.classList.remove('hidden');

        // B. Update step-by-step images
        imgOriginal.src = data.normalized_url + '?t=' + new Date().getTime();
        imgBgRemoved.src = data.bg_removed_url + '?t=' + new Date().getTime();
        imgDetected.src = data.detected_url + '?t=' + new Date().getTime();

        // Reset to first tab
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
        document.querySelector('[data-tab="original"]').classList.add('active');
        document.getElementById('tab-original').classList.add('active');

        // C. Update layout classification state
        if (data.is_valid_pan) {
            statusCard.classList.remove('invalid');
            statusIcon.innerHTML = '✓';
            statusTitle.innerHTML = 'Valid PAN Card Verified';
            statusDescription.innerHTML = 'Document structural integrity and alphanumeric records authenticated.';
        } else {
            statusCard.classList.add('invalid');
            statusIcon.innerHTML = '✗';
            statusTitle.innerHTML = 'Structural Validation Mismatch';
            statusDescription.innerHTML = 'The document layout does not match an official template format.';
        }

        // D. Populate extracted OCR fields
        fieldPan.value = data.ocr.pan_number || 'NOT DETECTED';
        fieldName.value = data.ocr.name || 'NOT DETECTED';
        fieldFather.value = data.ocr.father_name || 'NOT DETECTED';
        fieldDob.value = data.ocr.dob || 'NOT DETECTED';

        // E. Animate the forensic gauge scores
        const similarityPct = (data.layout_similarity * 100).toFixed(1);
        valCosine.innerHTML = similarityPct + '%';
        fillCosine.style.width = similarityPct + '%';

        valSsim.innerHTML = data.ssim_score.toFixed(4);
        fillSsim.style.width = (data.ssim_score * 100).toFixed(0) + '%';

        valBrightness.innerHTML = `Original: <strong>${data.brightness.original.toFixed(1)}</strong> | Isolated: <strong>${data.brightness.processed.toFixed(1)}</strong>`;

        // Clear and create element badges
        badgeRow.innerHTML = '';
        if (data.detected_classes.length === 0) {
            badgeRow.innerHTML = '<span class="text-muted">No structural elements isolated</span>';
        } else {
            data.detected_classes.forEach(cls => {
                const badge = document.createElement('span');
                badge.className = 'badge-anchor';
                badge.innerHTML = cls.toUpperCase();
                badgeRow.appendChild(badge);
            });
        }

        // F. Render Raw JSON Payload block
        jsonCode.innerHTML = JSON.stringify(data, null, 2);
    }

    /* ==========================================
       4. Visual Tab Switches
       ========================================== */
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Switch tabs active classes
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById('tab-' + targetTab).classList.add('active');
        });
    });

    /* ==========================================
       5. Clipboard Copy System
       ========================================== */
    btnCopyJson.addEventListener('click', () => {
        navigator.clipboard.writeText(jsonCode.innerHTML).then(() => {
            btnCopyJson.innerHTML = 'Copied!';
            btnCopyJson.style.backgroundColor = 'var(--color-green)';
            btnCopyJson.style.borderColor = 'var(--color-green)';
            
            setTimeout(() => {
                btnCopyJson.innerHTML = 'Copy Payload';
                btnCopyJson.style.backgroundColor = 'var(--panel-border)';
                btnCopyJson.style.borderColor = 'var(--panel-border)';
            }, 2000);
        });
    });
});
