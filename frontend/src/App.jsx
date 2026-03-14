import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("resnet18");
  const [gridSize, setGridSize] = useState(3);
  const [topk, setTopk] = useState(5);
  const [fillMode, setFillMode] = useState("constant");
  const [constantValue, setConstantValue] = useState(0);
  const [noiseStd, setNoiseStd] = useState(0.15);
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then((response) => response.json())
      .then((data) => {
        setModels(data.models || []);
        if (data.models?.length) {
          setSelectedModel(data.models[0]);
        }
      })
      .catch(() => {
        setError("Could not load available models. Make sure the FastAPI backend is running.");
      });
  }, []);

  useEffect(() => {
    if (!imageFile) {
      setPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(imageFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [imageFile]);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!imageFile) {
      setError("Choose an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("model_name", selectedModel);
    formData.append("grid_size", String(gridSize));
    formData.append("topk", String(topk));
    formData.append("fill_mode", fillMode);
    formData.append("constant_value", String(constantValue));
    formData.append("noise_std", String(noiseStd));

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed.");
      }

      setResults(data);
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-shell">
      <header className="hero">
        <p className="eyebrow">XAI-Vision</p>
        <h1>Inspect what a pretrained vision model actually relies on.</h1>
        <p className="hero-copy">
          Upload an image, choose a backbone, and rank the regions that most reduce the original prediction confidence when masked.
        </p>
      </header>

      <main className="layout">
        <section className="panel controls">
          <h2>Run Analysis</h2>
          <form onSubmit={handleSubmit}>
            <label>
              Image
              <input
                type="file"
                accept=".png,.jpg,.jpeg"
                onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
              />
            </label>

            <label>
              Model
              <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                {models.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Grid Size
              <input
                type="range"
                min="2"
                max="6"
                value={gridSize}
                onChange={(event) => setGridSize(Number(event.target.value))}
              />
              <span>{gridSize}</span>
            </label>

            <label>
              Top Regions
              <input
                type="range"
                min="1"
                max="10"
                value={topk}
                onChange={(event) => setTopk(Number(event.target.value))}
              />
              <span>{topk}</span>
            </label>

            <label>
              Fill Mode
              <select value={fillMode} onChange={(event) => setFillMode(event.target.value)}>
                <option value="constant">constant</option>
                <option value="mean">mean</option>
                <option value="noise">noise</option>
              </select>
            </label>

            <label>
              Constant Fill
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={constantValue}
                onChange={(event) => setConstantValue(Number(event.target.value))}
              />
              <span>{constantValue.toFixed(2)}</span>
            </label>

            <label>
              Noise Std
              <input
                type="range"
                min="0.01"
                max="0.5"
                step="0.01"
                value={noiseStd}
                onChange={(event) => setNoiseStd(Number(event.target.value))}
              />
              <span>{noiseStd.toFixed(2)}</span>
            </label>

            <button type="submit" disabled={loading}>
              {loading ? "Analyzing..." : "Analyze Image"}
            </button>
          </form>
        </section>

        <section className="panel preview">
          <h2>Input</h2>
          {previewUrl ? <img src={previewUrl} alt="Uploaded preview" /> : <p>No image selected yet.</p>}
        </section>

        <section className="panel results">
          <h2>Results</h2>
          {error ? <p className="error">{error}</p> : null}
          {!results ? <p>Run the analysis to see prediction and region importance.</p> : null}
          {results ? (
            <>
              <div className="metrics">
                <div>
                  <span className="metric-label">Prediction</span>
                  <strong>{results.class_name}</strong>
                </div>
                <div>
                  <span className="metric-label">Confidence</span>
                  <strong>{results.confidence.toFixed(4)}</strong>
                </div>
                <div>
                  <span className="metric-label">Device</span>
                  <strong>{results.device}</strong>
                </div>
              </div>

              <div className="image-grid">
                <figure>
                  <img src={`data:image/png;base64,${results.analysis_image}`} alt="Analyzed input" />
                  <figcaption>Analyzed View</figcaption>
                </figure>
                <figure>
                  <img src={`data:image/png;base64,${results.overlay_image}`} alt="Importance overlay" />
                  <figcaption>Importance Overlay</figcaption>
                </figure>
              </div>

              <div className="region-list">
                {results.top_regions.map((region) => (
                  <article key={region.rank} className="region-item">
                    <div>
                      <strong>#{region.rank}</strong>
                      <span>
                        bbox=(y:{region.bbox.y1}-{region.bbox.y2}, x:{region.bbox.x1}-{region.bbox.x2})
                      </span>
                    </div>
                    <strong>{region.importance.toFixed(4)}</strong>
                  </article>
                ))}
              </div>
            </>
          ) : null}
        </section>
      </main>
    </div>
  );
}

export default App;
