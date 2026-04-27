import { useState } from 'react';

const SuspectVideoUploader = () => {
  const [file, setFile] = useState(null);
  const [uploaderName, setUploaderName] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState({ message: '', type: '' });
  const [uploaderEmail, setUploaderEmail] = useState('');

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!file || !uploaderName || !uploaderEmail) {
      setStatus({ message: 'Please provide a file and source name.', type: 'error' });
      return;
    }

    setIsUploading(true);
    setStatus({ message: '', type: '' });

    const formData = new FormData();
    formData.append('file', file);
    formData.append('uploader_name', uploaderName);
    formData.append('uploader_email', uploaderEmail);
    formData.append('upload_source', 'dashboard-simulation');

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await fetch(`${API_URL}/upload-suspect`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setStatus({
          message: `🚨 Suspect video deployed to S3 as ${data.filename}. Watch your worker terminal!`,
          type: 'error' // Using the red 'error' style to make it look like an alert
        });
        setFile(null);
        setUploaderName('');
        setUploaderEmail('');

        document.getElementById('suspect-file-upload').value = '';
      } else {
        setStatus({ message: `❌ Error: ${data.detail}`, type: 'error' });
      }
    } catch (error) {
      setStatus({ message: `❌ Network error. Backend offline?`, type: 'error' });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="uploader-card suspect-card">
      <div className="uploader-header">
        <h2>Inject Suspect Video</h2>
        <p>Upload an unverified video to trigger the SSCD detection pipeline.</p>
      </div>

      <form onSubmit={handleUpload} className="uploader-form">
        <div className="form-group">
          <label htmlFor="suspectUploaderName">Simulated Source / Scraper Name</label>
          <input
            id="suspectUploaderName"
            type="text"
            value={uploaderName}
            onChange={(e) => setUploaderName(e.target.value)}
            placeholder="e.g., WebScraper_Bot_01"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="suspectUploaderEmail">Reporter Email</label>
          <input
            id="suspectUploaderEmail"
            type="email"
            value={uploaderEmail}
            onChange={(e) => setUploaderEmail(e.target.value)}
            placeholder="bot@scraper-system.net"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="suspect-file-upload">Suspect Video File</label>
          <input
            id="suspect-file-upload"
            type="file"
            accept="video/mp4,video/quicktime,video/x-matroska"
            onChange={handleFileChange}
            required
            className="file-input"
          />
        </div>

        <button
          type="submit"
          disabled={isUploading}
          className={`submit-btn suspect-btn ${isUploading ? 'loading' : ''}`}
        >
          {isUploading ? 'Injecting into AWS...' : 'Inject Suspect Video'}
        </button>
      </form>

      {status.message && (
        <div className={`status-message ${status.type}`}>
          {status.message}
        </div>
      )}
    </div>
  );
};

export default SuspectVideoUploader;