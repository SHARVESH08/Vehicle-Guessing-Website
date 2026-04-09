document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Particles.js
    if(typeof particlesJS !== 'undefined') {
        particlesJS("particles-js", {
            "particles": {
                "number": { "value": 60, "density": { "enable": true, "value_area": 800 } },
                "color": { "value": "#38bdf8" },
                "shape": { "type": "circle" },
                "opacity": { "value": 0.8, "random": true },
                "size": { "value": 5, "random": true },
                "line_linked": { "enable": true, "distance": 150, "color": "#818cf8", "opacity": 0.6, "width": 2 },
                "move": { "enable": true, "speed": 1, "direction": "none", "random": true, "out_mode": "out" }
            },
            "interactivity": {
                "detect_on": "window",
                "events": {
                    "onhover": { "enable": true, "mode": "grab" },
                    "onclick": { "enable": true, "mode": "push" },
                    "resize": true
                },
                "modes": {
                    "grab": { "distance": 140, "line_linked": { "opacity": 0.5 } },
                    "push": { "particles_nb": 3 }
                }
            },
            "retina_detect": true
        });
    }

    // 2. Fetch Metrics and Animate Counters
    fetch('/api/metrics')
        .then(res => res.json())
        .then(data => {
            if(!data.error) {
                animateCounter('m-acc', data.accuracy * 100);
                animateCounter('m-top3', data.top3_accuracy * 100);
                animateCounter('m-f1', data.f1_score * 100);
                animateCounter('m-roc', data.roc_auc, false);
                animateCounter('m-prec', data.precision * 100);
                animateCounter('m-rec', data.recall * 100);
                animateCounter('m-kappa', data.cohens_kappa, false);
                animateCounter('m-mcc', data.mcc, false);
                animateCounter('m-loss', data.log_loss, false);
                animateCounter('m-err', data.error_rate * 100);
            }
        })
        .catch(err => console.error("Error loading metrics:", err));

    function animateCounter(id, target, isPercentage=true) {
        let current = 0;
        const duration = 2000; // ms
        const steps = 60;
        const stepTime = Math.abs(Math.floor(duration / steps));
        const increment = target / steps;
        
        let timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            document.getElementById(id).innerText = isPercentage ? current.toFixed(1) : current.toFixed(3);
        }, stepTime);
    }

    // 3. Tab Navigation
    const navBtns = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.view-section');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const targetId = btn.getAttribute('data-target');
            sections.forEach(sec => {
                if(sec.id === targetId) {
                    sec.classList.add('active');
                } else {
                    sec.classList.remove('active');
                }
            });
        });
    });

    // 4. File Upload & Live Prediction Logic
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const resultZone = document.getElementById('result-zone');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('result-content');
    const resetBtn = document.getElementById('reset-btn');

    // Drag and Drop Events
    uploadZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('dragover'), false);
    });

    uploadZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    }, false);

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    resetBtn.addEventListener('click', () => {
        resultZone.classList.add('hidden');
        uploadZone.style.display = 'block';
        fileInput.value = ''; // clear input
        document.getElementById('conf-fill').style.width = '0%';
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        uploadZone.style.display = 'none';
        resultZone.classList.remove('hidden');
        loader.classList.remove('hidden');
        resultContent.classList.add('hidden');

        const formData = new FormData();
        formData.append('image', file);

        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            const data = await response.json();
            if(!response.ok) throw new Error(data.error || "Server error");
            return data;
        })
        .then(data => {
            document.getElementById('gradcam-preview').src = data.image_data;
            document.getElementById('pred-name').innerText = data.prediction;
            
            // Typewriter effect for confidence
            animateCounter('pred-conf-txt', data.confidence, true);
            
            setTimeout(() => {
                document.getElementById('conf-fill').style.width = `${data.confidence}%`;
            }, 500);

            const list = document.getElementById('reasons-list');
            list.innerHTML = '';
            data.reasons.forEach((reason, index) => {
                setTimeout(() => {
                    const li = document.createElement('li');
                    li.innerText = `Region Rank #${index+1}: ${reason}`;
                    list.appendChild(li);
                }, index * 400); // Stagger text appearance
            });

            loader.classList.add('hidden');
            resultContent.classList.remove('hidden');
        })
        .catch(err => {
            alert(err.message);
            console.error(err);
            resetBtn.click();
        });
    }

    // 5. Lightbox Logic
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxClose = document.querySelector('.lightbox-close');

    // Attach click listeners to all zoomable images (even dynamically added ones)
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('zoomable') && e.target.tagName === 'IMG') {
            lightboxImg.src = e.target.src;
            lightbox.classList.remove('hidden');
        }
    });

    lightboxClose.addEventListener('click', () => {
        lightbox.classList.add('hidden');
    });

    lightbox.addEventListener('click', (e) => {
        if (e.target !== lightboxImg) {
            lightbox.classList.add('hidden');
        }
    });
});
