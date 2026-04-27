import { useState } from 'react';

const OfficialAssetUploader = () => {
  const [file, setFile] = useState(null);
  const [assetId, setAssetId] = useState('');
  const [uploaderName, setUploaderName] = useState('');
  const [uploaderEmail, setUploaderEmail] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState({ message: '', type: '' });

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    
    if (!file || !assetId || !uploaderName || !uploaderEmail) {
      setStatus({ message: 'Please fill out all fields.', type: 'error' });
      return;
    }

    setIsUploading(true);
    setStatus({ message: '', type: '' });

    const formData = new FormData();
    formData.append('file', file);
    formData.append('asset_id', assetId);
    formData.append('uploader_name', uploaderName);
    formData.append('uploader_email', uploaderEmail)

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await fetch(`${API_URL}/upload-official`, {
        method: 'POST',
        body: formData, 
      });

      const data = await response.json();

      if (response.ok) {
        setStatus({ 
          message: `✅ Success! Master asset '${data.asset_id}' registered to S3.`, 
          type: 'success' 
        });
        // Reset form on success
        setFile(null);
        setAssetId('');
        setUploaderName('');
        setUploaderEmail('');
        // Reset the file input element visually
        document.getElementById('file-upload').value = '';
      } else {
        setStatus({ message: `❌ Error: ${data.detail}`, type: 'error' });
      }
    } catch (error) {
      setStatus({ message: `❌ Network error. Is the FastAPI server running?`, type: 'error' });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="uploader-card">
      <div className="uploader-header">
        <h2>Register Master Video</h2>
        <p>Upload official references to the secure S3 bucket for SSCD indexing.</p>
      </div>
      
      <form onSubmit={handleUpload} className="uploader-form">
        <div className="form-group">
          <label htmlFor="assetId">Asset ID (e.g., broadcast_001)</label>
          <input 
            id="assetId"
            type="text" 
            value={assetId} 
            onChange={(e) => setAssetId(e.target.value)} 
            placeholder="Enter unique asset identifier"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="uploaderName">Rights Holder / Uploader</label>
          <input 
            id="uploaderName"
            type="text" 
            value={uploaderName} 
            onChange={(e) => setUploaderName(e.target.value)} 
            placeholder="e.g., Network Admin"
            required
          />
        </div>
        
        <div className="form-group">
            <label htmlFor="officialUploaderEmail">Contact Email</label>
            <input 
                id="officialUploaderEmail"
                type="email" 
                value={uploaderEmail} 
                onChange={(e) => setUploaderEmail(e.target.value)} 
                placeholder="admin@studio.com"
                required
            />
        </div>

        <div className="form-group">
          <label htmlFor="file-upload">Master Video File</label>
          <input 
            id="file-upload"
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
          className={`submit-btn ${isUploading ? 'loading' : ''}`}
        >
          {isUploading ? 'Registering with AWS S3...' : 'Register Asset'}
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

export default OfficialAssetUploader;