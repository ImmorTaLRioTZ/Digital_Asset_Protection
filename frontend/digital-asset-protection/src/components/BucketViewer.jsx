import { useState, useEffect } from 'react';

const BucketViewer = () => {
  const [videos, setVideos] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchVideos = async () => {
    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await fetch(`${API_URL}/api/videos`);
      if (!response.ok) throw new Error('Failed to fetch bucket contents');

      const data = await response.json();
      setVideos(data.videos);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch on load, and auto-refresh every 10 seconds
  useEffect(() => {
    fetchVideos();
    const interval = setInterval(fetchVideos, 10000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading && videos.length === 0) return <div className="viewer-msg">Loading bucket data...</div>;
  if (error) return <div className="viewer-msg error">Error: {error}</div>;

  return (
    <div className="bucket-viewer">
      <div className="viewer-header">
        <h2>Live AWS S3 Bucket Monitor</h2>
        <button onClick={fetchVideos} className="refresh-btn">🔄 Refresh Now</button>
      </div>

      <div className="video-grid">
        {videos.map((vid) => (
          <div key={vid.id} className={`video-card ${vid.is_dmca_struck ? 'dmca-card' : ''} ${vid.asset_type === 'official-reference' ? 'official-card' : ''}`}>

            {/* Visual Badges */}
            <div className="badges">
              {vid.asset_type === 'official-reference' && <span className="badge official">Master Reference</span>}
              {vid.asset_type === 'suspect' && !vid.is_dmca_struck && <span className="badge suspect">Pending Scan</span>}
              {vid.is_dmca_struck && <span className="badge dmca">🚨 DMCA STRUCK</span>}
            </div>

            <div className="video-info">
              <h3 title={vid.filename}>{vid.filename.substring(0, 25)}...</h3>
              <p><strong>Uploader:</strong> {vid.uploader_name}</p>
              <p><strong>Date:</strong> {vid.date}</p>

              <a href={vid.url} target="_blank" rel="noreferrer" className="view-link">
                View Source File ↗
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BucketViewer;