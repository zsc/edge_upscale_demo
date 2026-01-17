document.addEventListener('DOMContentLoaded', () => {
    const processBtn = document.getElementById('processBtn');
    const imageInput = document.getElementById('imageInput');
    const resultsContainer = document.getElementById('resultsContainer');
    
    // Auto Canny toggle logic
    const autoCannyCheck = document.getElementById('auto_canny');
    const manualCannyGroup = document.getElementById('manual_canny_group');
    
    autoCannyCheck.addEventListener('change', (e) => {
        if (e.target.checked) {
            manualCannyGroup.style.display = 'none';
        } else {
            manualCannyGroup.style.display = 'flex';
        }
    });

    processBtn.addEventListener('click', async () => {
        const file = imageInput.files[0];
        if (!file) {
            alert("Please select an image first.");
            return;
        }

        processBtn.disabled = true;
        processBtn.textContent = "Processing...";
        resultsContainer.innerHTML = '<div style="text-align:center; width:100%;">Processing...</div>';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('scale', document.getElementById('scale').value);
        formData.append('edge_width_hr', document.getElementById('edge_width_hr').value);
        formData.append('auto_canny', document.getElementById('auto_canny').checked);
        formData.append('canny_low', document.getElementById('canny_low').value);
        formData.append('canny_high', document.getElementById('canny_high').value);
        formData.append('blur_sigma', document.getElementById('blur_sigma').value);
        formData.append('thinning', document.getElementById('thinning').checked);
        
        // Methods to run
        formData.append('methods', 'baseline,sdf,pde');

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            console.error(error);
            resultsContainer.innerHTML = `<div style="color:red; text-align:center;">Error: ${error.message}</div>`;
        } finally {
            processBtn.disabled = false;
            processBtn.textContent = "Run Upscale";
        }
    });

    function renderResults(data) {
        resultsContainer.innerHTML = '';
        const imgs = data.images;
        const meta = data.meta;
        
        // Helper to create card
        const createCard = (title, b64, isEdge = true) => {
            const div = document.createElement('div');
            div.className = 'result-card';
            
            const imgClass = isEdge ? 'edge-img' : '';
            
            div.innerHTML = `
                <h3>${title}</h3>
                <div class="img-container">
                    <img src="${b64}" class="${imgClass}" alt="${title}">
                </div>
                <a href="${b64}" download="${title.toLowerCase().replace(/ /g, '_')}.png" class="download-btn">Download PNG</a>
            `;
            return div;
        };

        // 1. Original (Not pixelated usually, but kept consistent or raw)
        resultsContainer.appendChild(createCard(`Original (${meta.input_size[1]}x${meta.input_size[0]})`, imgs.original, false));

        // 2. LR Edge
        resultsContainer.appendChild(createCard(`LR Edge Map`, imgs.edge_lr, true));

        // 3. Baseline
        if (imgs.baseline)
            resultsContainer.appendChild(createCard(`Baseline (Bicubic)`, imgs.baseline, true));

        // 4. SDF
        if (imgs.sdf)
            resultsContainer.appendChild(createCard(`SDF Method`, imgs.sdf, true));
            
        // 5. PDE
        if (imgs.pde)
            resultsContainer.appendChild(createCard(`PDE (Shock Filter)`, imgs.pde, true));
    }
});
