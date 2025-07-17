// NeuroInsight Media Analysis Platform
class MediaAnalysisPlatform {
    constructor() {
        this.uploadedFiles = {
            text: [],
            images: [],
            videos: [],
            audio: [],
            urls: []
        };
        
        this.init();
    }

    init() {
        this.setupTextTabs();
        this.setupTextMetrics();
        this.setupFileUploads();
        this.setupUrlAnalysis();
        this.setupActionHandlers();
        this.updateContentCounts();
    }

    // Text Input Functionality
    setupTextTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove active class from all tabs
                tabBtns.forEach(b => b.classList.remove('active'));
                tabContents.forEach(c => c.classList.remove('active'));

                // Add active class to clicked tab
                btn.classList.add('active');
                const tabId = btn.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
            });
        });
    }

    setupTextMetrics() {
        const textareas = document.querySelectorAll('textarea');
        
        textareas.forEach(textarea => {
            textarea.addEventListener('input', (e) => {
                this.updateTextMetrics(e.target);
            });
        });
    }

    updateTextMetrics(textarea) {
        const text = textarea.value;
        const parentTab = textarea.closest('.tab-content');
        const metricsDiv = parentTab.querySelector('.text-metrics');
        
        const charCount = text.length;
        const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
        const hashtagCount = (text.match(/#\w+/g) || []).length;
        
        // Update character count
        const charCountSpan = metricsDiv.querySelector('.character-count');
        if (charCountSpan) charCountSpan.textContent = `${charCount} characters`;
        
        // Update word count
        const wordCountSpan = metricsDiv.querySelector('.word-count');
        if (wordCountSpan) wordCountSpan.textContent = `${wordCount} words`;
        
        // Update hashtag count
        const hashtagCountSpan = metricsDiv.querySelector('.hashtag-count');
        if (hashtagCountSpan) hashtagCountSpan.textContent = `${hashtagCount} hashtags`;
        
        // Calculate readability score (simplified)
        const readabilitySpan = metricsDiv.querySelector('.readability-score');
        if (readabilitySpan && wordCount > 0) {
            const avgWordsPerSentence = wordCount / (text.split(/[.!?]+/).length - 1 || 1);
            const readabilityScore = this.calculateReadabilityScore(avgWordsPerSentence, text);
            readabilitySpan.textContent = `Readability: ${readabilityScore}`;
        }

        // Store text content
        this.uploadedFiles.text = this.getAllTextContent();
        this.updateContentCounts();
    }

    calculateReadabilityScore(avgWordsPerSentence, text) {
        if (avgWordsPerSentence < 15) return 'Easy';
        if (avgWordsPerSentence < 20) return 'Medium';
        return 'Complex';
    }

    getAllTextContent() {
        const textareas = document.querySelectorAll('textarea');
        const textContent = [];
        
        textareas.forEach(textarea => {
            if (textarea.value.trim()) {
                textContent.push({
                    type: textarea.id,
                    content: textarea.value.trim(),
                    wordCount: textarea.value.trim().split(/\s+/).length,
                    charCount: textarea.value.length
                });
            }
        });
        
        return textContent;
    }

    // File Upload Functionality
    setupFileUploads() {
        this.setupImageUpload();
        this.setupVideoUpload();
        this.setupAudioUpload();
    }

    setupImageUpload() {
        const dropZone = document.getElementById('image-drop-zone');
        const fileInput = document.getElementById('image-file-input');
        const previewContainer = document.getElementById('image-preview-container');

        this.setupDropZone(dropZone, fileInput, (files) => {
            this.handleImageFiles(files, previewContainer);
        });

        fileInput.addEventListener('change', (e) => {
            this.handleImageFiles(e.target.files, previewContainer);
        });
    }

    setupVideoUpload() {
        const dropZone = document.getElementById('video-drop-zone');
        const fileInput = document.getElementById('video-file-input');
        const previewContainer = document.getElementById('video-preview-container');

        this.setupDropZone(dropZone, fileInput, (files) => {
            this.handleVideoFiles(files, previewContainer);
        });

        fileInput.addEventListener('change', (e) => {
            this.handleVideoFiles(e.target.files, previewContainer);
        });
    }

    setupAudioUpload() {
        const dropZone = document.getElementById('audio-drop-zone');
        const fileInput = document.getElementById('audio-file-input');
        const previewContainer = document.getElementById('audio-preview-container');

        this.setupDropZone(dropZone, fileInput, (files) => {
            this.handleAudioFiles(files, previewContainer);
        });

        fileInput.addEventListener('change', (e) => {
            this.handleAudioFiles(e.target.files, previewContainer);
        });
    }

    setupDropZone(dropZone, fileInput, handleFiles) {
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });
    }

    handleImageFiles(files, container) {
        Array.from(files).forEach(file => {
            if (this.isValidImageFile(file)) {
                this.uploadedFiles.images.push(file);
                this.createImagePreview(file, container);
            } else {
                this.showError(`Invalid image file: ${file.name}`);
            }
        });
        this.updateContentCounts();
    }

    handleVideoFiles(files, container) {
        Array.from(files).forEach(file => {
            if (this.isValidVideoFile(file)) {
                this.uploadedFiles.videos.push(file);
                this.createVideoPreview(file, container);
            } else {
                this.showError(`Invalid video file: ${file.name}`);
            }
        });
        this.updateContentCounts();
    }

    handleAudioFiles(files, container) {
        Array.from(files).forEach(file => {
            if (this.isValidAudioFile(file)) {
                this.uploadedFiles.audio.push(file);
                this.createAudioPreview(file, container);
            } else {
                this.showError(`Invalid audio file: ${file.name}`);
            }
        });
        this.updateContentCounts();
    }

    isValidImageFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'];
        return validTypes.includes(file.type) && file.size < 10 * 1024 * 1024; // 10MB limit
    }

    isValidVideoFile(file) {
        const validTypes = ['video/mp4', 'video/mov', 'video/quicktime', 'video/avi', 'video/webm'];
        return validTypes.includes(file.type) && file.size < 100 * 1024 * 1024; // 100MB limit
    }

    isValidAudioFile(file) {
        const validTypes = ['audio/mpeg', 'audio/wav', 'audio/aac', 'audio/ogg'];
        return validTypes.includes(file.type) && file.size < 50 * 1024 * 1024; // 50MB limit
    }

    createImagePreview(file, container) {
        const preview = document.createElement('div');
        preview.className = 'file-preview';
        
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.alt = file.name;
        
        const fileInfo = document.createElement('div');
        fileInfo.className = 'file-info';
        fileInfo.innerHTML = `
            <div class="file-name">${file.name}</div>
            <div class="file-size">${this.formatFileSize(file.size)}</div>
        `;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = () => {
            this.removeFile(file, 'images', preview);
        };
        
        preview.appendChild(img);
        preview.appendChild(fileInfo);
        preview.appendChild(removeBtn);
        container.appendChild(preview);
    }

    createVideoPreview(file, container) {
        const preview = document.createElement('div');
        preview.className = 'file-preview';
        
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.controls = true;
        video.muted = true;
        
        const fileInfo = document.createElement('div');
        fileInfo.className = 'file-info';
        fileInfo.innerHTML = `
            <div class="file-name">${file.name}</div>
            <div class="file-size">${this.formatFileSize(file.size)}</div>
        `;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = () => {
            this.removeFile(file, 'videos', preview);
        };
        
        preview.appendChild(video);
        preview.appendChild(fileInfo);
        preview.appendChild(removeBtn);
        container.appendChild(preview);
    }

    createAudioPreview(file, container) {
        const preview = document.createElement('div');
        preview.className = 'audio-preview';
        
        const audioIcon = document.createElement('div');
        audioIcon.className = 'audio-icon';
        audioIcon.textContent = '🎵';
        
        const audio = document.createElement('audio');
        audio.src = URL.createObjectURL(file);
        audio.controls = true;
        
        const fileInfo = document.createElement('div');
        fileInfo.className = 'file-info';
        fileInfo.innerHTML = `
            <div class="file-name">${file.name}</div>
            <div class="file-size">${this.formatFileSize(file.size)}</div>
        `;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = () => {
            this.removeFile(file, 'audio', preview);
        };
        
        preview.appendChild(audioIcon);
        preview.appendChild(audio);
        preview.appendChild(fileInfo);
        preview.appendChild(removeBtn);
        container.appendChild(preview);
    }

    removeFile(file, type, previewElement) {
        const index = this.uploadedFiles[type].indexOf(file);
        if (index > -1) {
            this.uploadedFiles[type].splice(index, 1);
        }
        previewElement.remove();
        this.updateContentCounts();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // URL Analysis Functionality
    setupUrlAnalysis() {
        const urlInput = document.getElementById('url-input');
        const captureBtn = document.getElementById('capture-url');
        const previewContainer = document.getElementById('url-preview-container');

        captureBtn.addEventListener('click', () => {
            const url = urlInput.value.trim();
            if (this.isValidUrl(url)) {
                this.captureUrl(url, previewContainer);
            } else {
                this.showError('Please enter a valid URL');
            }
        });

        urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                captureBtn.click();
            }
        });
    }

    isValidUrl(string) {
        try {
            new URL(string);
            return true;
        } catch (_) {
            return false;
        }
    }

    captureUrl(url, container) {
        // Simulate URL capture process
        this.showLoading('Capturing website...');
        
        setTimeout(() => {
            this.uploadedFiles.urls.push({
                url: url,
                timestamp: new Date(),
                title: this.extractDomainName(url)
            });
            
            this.createUrlPreview(url, container);
            this.hideLoading();
            this.updateContentCounts();
            
            // Clear input
            document.getElementById('url-input').value = '';
        }, 2000);
    }

    extractDomainName(url) {
        try {
            return new URL(url).hostname;
        } catch (e) {
            return 'Unknown Website';
        }
    }

    createUrlPreview(url, container) {
        const preview = document.createElement('div');
        preview.className = 'url-preview-item';
        
        // Create a placeholder screenshot
        const screenshot = document.createElement('div');
        screenshot.style.width = '100%';
        screenshot.style.height = '200px';
        screenshot.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
        screenshot.style.borderRadius = '4px';
        screenshot.style.display = 'flex';
        screenshot.style.alignItems = 'center';
        screenshot.style.justifyContent = 'center';
        screenshot.style.color = 'white';
        screenshot.style.fontSize = '1.2rem';
        screenshot.style.fontWeight = 'bold';
        screenshot.textContent = 'Website Preview';
        
        const urlInfo = document.createElement('div');
        urlInfo.className = 'url-info';
        urlInfo.innerHTML = `
            <div class="url-title">${this.extractDomainName(url)}</div>
            <div class="url-link">${url}</div>
            <div class="capture-time">Captured: ${new Date().toLocaleString()}</div>
        `;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = () => {
            const index = this.uploadedFiles.urls.findIndex(item => item.url === url);
            if (index > -1) {
                this.uploadedFiles.urls.splice(index, 1);
            }
            preview.remove();
            this.updateContentCounts();
        };
        
        preview.appendChild(screenshot);
        preview.appendChild(urlInfo);
        preview.appendChild(removeBtn);
        container.appendChild(preview);
    }

    // Action Handlers
    setupActionHandlers() {
        // Text actions
        document.getElementById('clear-text').addEventListener('click', () => {
            const activeTab = document.querySelector('.tab-content.active');
            const textarea = activeTab.querySelector('textarea');
            textarea.value = '';
            this.updateTextMetrics(textarea);
        });

        document.getElementById('paste-clipboard').addEventListener('click', async () => {
            try {
                const text = await navigator.clipboard.readText();
                const activeTab = document.querySelector('.tab-content.active');
                const textarea = activeTab.querySelector('textarea');
                textarea.value = text;
                this.updateTextMetrics(textarea);
                this.showSuccess('Text pasted from clipboard');
            } catch (err) {
                this.showError('Could not access clipboard');
            }
        });

        // Clear actions
        document.getElementById('clear-images').addEventListener('click', () => {
            this.clearFiles('images', 'image-preview-container');
        });

        document.getElementById('clear-videos').addEventListener('click', () => {
            this.clearFiles('videos', 'video-preview-container');
        });

        document.getElementById('clear-audio').addEventListener('click', () => {
            this.clearFiles('audio', 'audio-preview-container');
        });

        // Analysis actions
        document.getElementById('analyze-text').addEventListener('click', () => {
            this.analyzeContent('text');
        });

        document.getElementById('analyze-images').addEventListener('click', () => {
            this.analyzeContent('images');
        });

        document.getElementById('analyze-videos').addEventListener('click', () => {
            this.analyzeContent('videos');
        });

        document.getElementById('analyze-audio').addEventListener('click', () => {
            this.analyzeContent('audio');
        });

        document.getElementById('analyze-campaign').addEventListener('click', () => {
            this.analyzeContent('campaign');
        });

        // Modal actions
        document.getElementById('close-modal').addEventListener('click', () => {
            this.hideModal();
        });

        // Campaign actions
        document.getElementById('save-campaign').addEventListener('click', () => {
            this.saveCampaign();
        });

        document.getElementById('export-campaign').addEventListener('click', () => {
            this.exportCampaign();
        });
    }

    clearFiles(type, containerId) {
        this.uploadedFiles[type] = [];
        document.getElementById(containerId).innerHTML = '';
        this.updateContentCounts();
    }

    // Analysis Functions
    analyzeContent(type) {
        const contentCount = this.getContentCount(type);
        
        if (contentCount === 0) {
            this.showError(`No ${type} content to analyze`);
            return;
        }

        this.showLoading(`Analyzing ${type} content...`);
        
        // Simulate analysis process
        setTimeout(() => {
            const results = this.generateAnalysisResults(type);
            this.hideLoading();
            this.showAnalysisResults(results);
        }, 3000 + Math.random() * 2000);
    }

    getContentCount(type) {
        if (type === 'text') {
            return this.uploadedFiles.text.length;
        } else if (type === 'campaign') {
            return Object.values(this.uploadedFiles).reduce((total, arr) => total + arr.length, 0);
        } else {
            return this.uploadedFiles[type].length;
        }
    }

    generateAnalysisResults(type) {
        const baseResults = {
            type: type,
            timestamp: new Date().toLocaleString(),
            sentiment: {
                positive: Math.floor(Math.random() * 30) + 40,
                neutral: Math.floor(Math.random() * 30) + 20,
                negative: Math.floor(Math.random() * 20) + 10
            },
            engagement: {
                attention: Math.floor(Math.random() * 30) + 60,
                emotional_response: Math.floor(Math.random() * 25) + 55,
                memorability: Math.floor(Math.random() * 35) + 45
            },
            psychological_triggers: [
                'Trust & Authority',
                'Social Proof',
                'Urgency & Scarcity',
                'Emotional Appeal'
            ]
        };

        if (type === 'text') {
            baseResults.text_metrics = {
                readability: 'Medium',
                tone: 'Professional',
                word_count: this.uploadedFiles.text.reduce((total, item) => total + item.wordCount, 0),
                key_themes: ['Innovation', 'Quality', 'Reliability']
            };
        } else if (type === 'images') {
            baseResults.visual_metrics = {
                color_palette: ['#667eea', '#764ba2', '#ffffff'],
                composition: 'Balanced',
                visual_appeal: 82,
                brand_consistency: 'High'
            };
        } else if (type === 'videos') {
            baseResults.video_metrics = {
                duration: '2m 34s',
                engagement_curve: 'Rising',
                key_moments: ['0:15 - Product reveal', '1:30 - Call to action'],
                audio_sentiment: 'Positive'
            };
        } else if (type === 'campaign') {
            baseResults.campaign_metrics = {
                cross_media_consistency: 'High',
                message_alignment: 85,
                target_audience_match: 'Excellent',
                recommended_optimizations: ['Strengthen call-to-action', 'Enhance visual hierarchy']
            };
        }

        return baseResults;
    }

    showAnalysisResults(results) {
        const modal = document.getElementById('analysis-modal');
        const resultsContainer = document.getElementById('analysis-results');
        
        resultsContainer.innerHTML = `
            <div class="results-header">
                <h3>${results.type.charAt(0).toUpperCase() + results.type.slice(1)} Analysis Results</h3>
                <p class="analysis-timestamp">Generated: ${results.timestamp}</p>
            </div>
            
            <div class="results-grid">
                <div class="result-card">
                    <h4>Sentiment Analysis</h4>
                    <div class="sentiment-bars">
                        <div class="sentiment-bar">
                            <span>Positive</span>
                            <div class="bar"><div class="fill" style="width: ${results.sentiment.positive}%"></div></div>
                            <span>${results.sentiment.positive}%</span>
                        </div>
                        <div class="sentiment-bar">
                            <span>Neutral</span>
                            <div class="bar"><div class="fill" style="width: ${results.sentiment.neutral}%"></div></div>
                            <span>${results.sentiment.neutral}%</span>
                        </div>
                        <div class="sentiment-bar">
                            <span>Negative</span>
                            <div class="bar"><div class="fill" style="width: ${results.sentiment.negative}%"></div></div>
                            <span>${results.sentiment.negative}%</span>
                        </div>
                    </div>
                </div>
                
                <div class="result-card">
                    <h4>Engagement Metrics</h4>
                    <div class="engagement-metrics">
                        <div class="metric">
                            <span class="metric-label">Attention Score</span>
                            <span class="metric-value">${results.engagement.attention}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Emotional Response</span>
                            <span class="metric-value">${results.engagement.emotional_response}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Memorability</span>
                            <span class="metric-value">${results.engagement.memorability}%</span>
                        </div>
                    </div>
                </div>
                
                <div class="result-card">
                    <h4>Psychological Triggers</h4>
                    <div class="triggers-list">
                        ${results.psychological_triggers.map(trigger => `<span class="trigger-tag">${trigger}</span>`).join('')}
                    </div>
                </div>
            </div>
            
            <div class="results-actions">
                <button class="action-btn secondary" onclick="platform.exportResults()">Export Results</button>
                <button class="action-btn primary" onclick="platform.generateRecommendations()">Get Recommendations</button>
            </div>
        `;
        
        // Add custom styles for results
        if (!document.getElementById('results-styles')) {
            const styles = document.createElement('style');
            styles.id = 'results-styles';
            styles.textContent = `
                .results-header { margin-bottom: 2rem; }
                .analysis-timestamp { color: #666; font-size: 0.9rem; }
                .results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
                .result-card { padding: 1.5rem; background: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef; }
                .result-card h4 { margin-bottom: 1rem; color: #333; }
                .sentiment-bars { display: flex; flex-direction: column; gap: 0.75rem; }
                .sentiment-bar { display: flex; align-items: center; gap: 1rem; }
                .bar { flex: 1; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }
                .fill { height: 100%; background: #667eea; transition: width 0.5s; }
                .engagement-metrics { display: flex; flex-direction: column; gap: 1rem; }
                .metric { display: flex; justify-content: space-between; align-items: center; }
                .metric-value { font-weight: bold; color: #667eea; }
                .triggers-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
                .trigger-tag { padding: 0.5rem 1rem; background: #667eea; color: white; border-radius: 20px; font-size: 0.9rem; }
                .results-actions { display: flex; gap: 1rem; justify-content: center; margin-top: 2rem; }
            `;
            document.head.appendChild(styles);
        }
        
        modal.classList.add('active');
    }

    hideModal() {
        document.getElementById('analysis-modal').classList.remove('active');
    }

    // Loading and Message Functions
    showLoading(message) {
        const overlay = document.getElementById('loading-overlay');
        const messageEl = document.getElementById('loading-message');
        const progressFill = document.getElementById('progress-fill');
        
        messageEl.textContent = message;
        overlay.classList.add('active');
        
        // Simulate progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 95) progress = 95;
            progressFill.style.width = progress + '%';
        }, 200);
        
        // Store interval to clear later
        this.progressInterval = interval;
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        const progressFill = document.getElementById('progress-fill');
        
        progressFill.style.width = '100%';
        
        setTimeout(() => {
            overlay.classList.remove('active');
            progressFill.style.width = '0%';
            if (this.progressInterval) {
                clearInterval(this.progressInterval);
            }
        }, 500);
    }

    showError(message) {
        // Simple error display - could be enhanced with a proper notification system
        alert('Error: ' + message);
    }

    showSuccess(message) {
        // Simple success display - could be enhanced with a proper notification system
        console.log('Success: ' + message);
    }

    // Content Count Updates
    updateContentCounts() {
        document.getElementById('text-count').textContent = this.uploadedFiles.text.length;
        document.getElementById('image-count').textContent = this.uploadedFiles.images.length;
        document.getElementById('video-count').textContent = this.uploadedFiles.videos.length;
        document.getElementById('audio-count').textContent = this.uploadedFiles.audio.length;
        document.getElementById('url-count').textContent = this.uploadedFiles.urls.length;
    }

    // Campaign Management
    saveCampaign() {
        const campaignData = {
            timestamp: new Date().toISOString(),
            content: this.uploadedFiles
        };
        
        localStorage.setItem('neuroinsight_campaign', JSON.stringify(campaignData));
        this.showSuccess('Campaign saved successfully');
    }

    exportCampaign() {
        const campaignData = {
            timestamp: new Date().toISOString(),
            content_summary: {
                text_inputs: this.uploadedFiles.text.length,
                images: this.uploadedFiles.images.length,
                videos: this.uploadedFiles.videos.length,
                audio_files: this.uploadedFiles.audio.length,
                urls: this.uploadedFiles.urls.length
            },
            text_content: this.uploadedFiles.text,
            urls: this.uploadedFiles.urls
        };
        
        const blob = new Blob([JSON.stringify(campaignData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `neuroinsight_campaign_${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    exportResults() {
        this.showSuccess('Results exported successfully');
    }

    generateRecommendations() {
        this.showSuccess('Generating recommendations...');
    }
}

// Initialize the platform when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.platform = new MediaAnalysisPlatform();
});