
import { useMemo, useState } from "react";
import "./App.css";

const initialIncidents = [
  {
    id: "VA-001",
    type: "Motorcycle fall",
    camera: "Camera 01",
    time: "20:00:20",
    severity: "High",
    status: "Needs review",
    location: "Highway Zone A",
  },
  {
    id: "VA-002",
    type: "Vehicle stopped",
    camera: "Camera 02",
    time: "20:02:10",
    severity: "Medium",
    status: "Needs review",
    location: "Highway Zone B",
  },
  {
    id: "VA-003",
    type: "Possible collision",
    camera: "Camera 03",
    time: "20:04:35",
    severity: "High",
    status: "Reviewed",
    location: "Highway Zone C",
  },
];

const navigation = [
  { icon: "▦", label: "Overview" },
  { icon: "▣", label: "CCTV Monitoring" },
  { icon: "⚠", label: "Incidents" },
  { icon: "⌖", label: "Locations" },
  { icon: "☷", label: "Reports" },
];

function App() {
  const [incidents, setIncidents] = useState(initialIncidents);
  const [activeNav, setActiveNav] = useState("Overview");
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null);

  const filteredIncidents = useMemo(() => {
    return incidents.filter((incident) => {
      const matchesFilter =
        filter === "All" || incident.status === filter;

      const text = [
        incident.id,
        incident.type,
        incident.camera,
        incident.location,
      ]
        .join(" ")
        .toLowerCase();

      return (
        matchesFilter &&
        text.includes(search.toLowerCase())
      );
    });
  }, [incidents, filter, search]);

  const pendingCount = incidents.filter(
    (item) => item.status === "Needs review"
  ).length;

  function markReviewed(id) {
    setIncidents((items) =>
      items.map((item) =>
        item.id === id
          ? { ...item, status: "Reviewed" }
          : item
      )
    );

    setSelected(null);
  }

  return (
    <div className="app-shell">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">V</div>

          <div>
            <h2>VITA AEGIS</h2>
            <span>ROAD SAFETY SYSTEM</span>
          </div>
        </div>

        <div className="nav-heading">WORKSPACE</div>

        <nav>
          {navigation.map((item) => (
            <button
              key={item.label}
              className={`nav-item ${
                activeNav === item.label ? "active" : ""
              }`}
              onClick={() => setActiveNav(item.label)}
            >
              <span className="nav-icon">{item.icon}</span>

              {item.label}

              {item.label === "Incidents" &&
                pendingCount > 0 && (
                  <span className="nav-count">
                    {pendingCount}
                  </span>
                )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-dot" />

          <div>
            <strong>Prototype running</strong>
            <span>Local demonstration</span>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">
              Vita Aegis / {activeNav}
            </div>
            <h1>{activeNav}</h1>
          </div>

          <div className="top-actions">
            <span className="demo-badge">
              ● DEMO MODE
            </span>
            <div className="avatar">VA</div>
          </div>
        </header>

        {/* DEMO NOTICE */}
        <div className="notice">
          <strong>DEMO DATA</strong>
          <span>
            Sample incident records and recorded accident footage.
            No live CCTV connection or emergency notifications
            are currently enabled.
          </span>
        </div>

        {/* OVERVIEW */}
        {activeNav === "Overview" ? (
          <>
            <section className="stats-grid">
              <StatCard
                label="Configured cameras"
                value="04"
                detail="Sample configuration"
                icon="▣"
                tone="blue"
              />

              <StatCard
                label="Sample incidents"
                value={String(incidents.length).padStart(2, "0")}
                detail="Illustrative records"
                icon="⚠"
                tone="orange"
              />

              <StatCard
                label="Awaiting review"
                value={String(pendingCount).padStart(2, "0")}
                detail="Manual review required"
                icon="◷"
                tone="purple"
              />

              <StatCard
                label="Detection engine"
                value="Demo"
                detail="Not connected"
                icon="⌁"
                tone="green"
              />
            </section>

            {/* CCTV AND SYSTEM STATUS */}
            <section className="content-grid">
              <div className="panel camera-panel">
                <div className="panel-heading">
                  <div>
                    <h3>CCTV Monitoring</h3>
                    <p>Recorded accident footage preview</p>
                  </div>

                  <span className="small-badge">
                    SAMPLE VIDEO
                  </span>
                </div>

                <div className="camera-screen">
                  <div className="camera-topline">
                    <span>
                      CAM-01 / HIGHWAY ZONE A
                    </span>

                    <span className="video-indicator">
                      ● RECORDED FOOTAGE
                    </span>
                  </div>

                  {/* ACCIDENT VIDEO */}
                  <div className="camera-video">
                    <video
                      controls
                      autoPlay
                      muted
                      loop
                      playsInline
                      preload="metadata"
                    >
                      <source
                        src="/accident_clip_web.mp4"
                        type="video/mp4"
                      />

                      Your browser does not support
                      video playback.
                    </video>
                  </div>

                  <div className="camera-bottomline">
                    <span>
                      Source: Sample accident video
                    </span>

                    <span>
                      AI: Analysis prototype
                    </span>
                  </div>
                </div>

                <div className="camera-footer">
                  <span>
                    <i className="legend-dot gray" />
                    Recorded footage
                  </span>

                  <span>
                    Camera 01 / 04
                  </span>
                </div>
              </div>

              {/* SYSTEM STATUS */}
              <div className="panel status-panel">
                <div className="panel-heading">
                  <div>
                    <h3>System Overview</h3>
                    <p>Current prototype status</p>
                  </div>
                </div>

                <div className="status-row">
                  <span className="status-symbol">
                    ▣
                  </span>

                  <div className="status-copy">
                    <strong>Camera connections</strong>
                    <span>
                      Recorded video available
                    </span>
                  </div>

                  <span className="status-pill green-pill">
                    Video Ready
                  </span>
                </div>

                <div className="status-row">
                  <span className="status-symbol">
                    ⌁
                  </span>

                  <div className="status-copy">
                    <strong>AI detection</strong>
                    <span>
                      Python analysis prototype
                    </span>
                  </div>

                  <span className="status-pill amber-pill">
                    Prototype
                  </span>
                </div>

                <div className="status-row">
                  <span className="status-symbol">
                    ♧
                  </span>

                  <div className="status-copy">
                    <strong>Emergency alerts</strong>
                    <span>
                      Notifications not connected
                    </span>
                  </div>

                  <span className="status-pill gray-pill">
                    Inactive
                  </span>
                </div>

                <div className="info-box">
                  <strong>Next milestone</strong>

                  <p>
                    Integrate Python accident-analysis results
                    with the dashboard to display verified
                    incident events.
                  </p>
                </div>
              </div>
            </section>

            {/* INCIDENT TABLE */}
            <IncidentTable
              incidents={filteredIncidents}
              filter={filter}
              setFilter={setFilter}
              search={search}
              setSearch={setSearch}
              onSelect={setSelected}
            />
          </>
        ) : activeNav === "Incidents" ? (
          <IncidentTable
            incidents={filteredIncidents}
            filter={filter}
            setFilter={setFilter}
            search={search}
            setSearch={setSearch}
            onSelect={setSelected}
          />
        ) : activeNav === "CCTV Monitoring" ? (
          <section className="panel camera-panel">
            <div className="panel-heading">
              <div>
                <h3>Highway Camera 01</h3>
                <p>Recorded accident footage</p>
              </div>

              <span className="small-badge">
                SAMPLE VIDEO
              </span>
            </div>

            <div className="camera-video standalone-video">
              <video
                controls
                autoPlay
                muted
                loop
                playsInline
                preload="metadata"
              >
                <source
                  src="/accident_clip_web.mp4"
                  type="video/mp4"
                />
                Your browser does not support video playback.
              </video>
            </div>

            <div className="info-box">
              This is prerecorded accident footage for
              demonstration. It is not a live CCTV stream.
            </div>
          </section>
        ) : (
          <section className="panel placeholder-page">
            <div className="camera-symbol">▦</div>

            <h2>{activeNav}</h2>

            <p>
              This section is part of the Vita Aegis
              dashboard prototype. Additional functionality
              will be integrated in future development.
            </p>

            <button
              className="primary-button"
              onClick={() => setActiveNav("Overview")}
            >
              Back to overview
            </button>
          </section>
        )}

        <footer>
          VITA AEGIS · Road Accident Monitoring Prototype
          <span>For demonstration only</span>
        </footer>
      </main>

      {/* INCIDENT DETAILS MODAL */}
      {selected && (
        <div
          className="modal-backdrop"
          onClick={() => setSelected(null)}
        >
          <section
            className="modal"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-heading">
              <div>
                <span className="breadcrumb">
                  INCIDENT DETAILS
                </span>

                <h2>{selected.id}</h2>
              </div>

              <button
                className="close-button"
                onClick={() => setSelected(null)}
              >
                ×
              </button>
            </div>

            <div className="notice">
              <strong>SAMPLE</strong>

              <span>
                This record is illustrative and is not
                a confirmed real-time accident alert.
              </span>
            </div>

            <div className="detail-list">
              <Detail
                label="Incident type"
                value={selected.type}
              />

              <Detail
                label="Camera"
                value={selected.camera}
              />

              <Detail
                label="Timestamp"
                value={selected.time}
              />

              <Detail
                label="Location"
                value={selected.location}
              />

              <Detail
                label="Severity"
                value={selected.severity}
              />

              <Detail
                label="Review status"
                value={selected.status}
              />
            </div>

            {selected.status !== "Reviewed" && (
              <button
                className="primary-button full-button"
                onClick={() => markReviewed(selected.id)}
              >
                Mark as reviewed
              </button>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

/* STATISTICS CARD */
function StatCard({
  label,
  value,
  detail,
  icon,
  tone,
}) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${tone}`}>
        {icon}
      </div>

      <div className="stat-label">{label}</div>

      <div className="stat-value">{value}</div>

      <div className="stat-detail">{detail}</div>
    </div>
  );
}

/* INCIDENT TABLE */
function IncidentTable({
  incidents,
  filter,
  setFilter,
  search,
  setSearch,
  onSelect,
}) {
  return (
    <section className="panel incidents-panel">
      <div className="panel-heading incident-heading">
        <div>
          <h3>Incident Records</h3>
          <p>Illustrative events for interface testing</p>
        </div>

        <div className="table-controls">
          <input
            aria-label="Search incidents"
            placeholder="Search incidents..."
            value={search}
            onChange={(event) =>
              setSearch(event.target.value)
            }
          />

          <select
            aria-label="Filter by status"
            value={filter}
            onChange={(event) =>
              setFilter(event.target.value)
            }
          >
            <option>All</option>
            <option>Needs review</option>
            <option>Reviewed</option>
          </select>
        </div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>INCIDENT</th>
              <th>TYPE</th>
              <th>CAMERA / TIME</th>
              <th>SEVERITY</th>
              <th>STATUS</th>
              <th>ACTION</th>
            </tr>
          </thead>

          <tbody>
            {incidents.length ? (
              incidents.map((incident) => (
                <tr key={incident.id}>
                  <td>
                    <strong>{incident.id}</strong>

                    <span className="subcell">
                      {incident.location}
                    </span>
                  </td>

                  <td>{incident.type}</td>

                  <td>
                    {incident.camera}

                    <span className="subcell">
                      {incident.time}
                    </span>
                  </td>

                  <td>
                    <span
                      className={`severity ${
                        incident.severity === "High"
                          ? "high"
                          : "medium"
                      }`}
                    >
                      {incident.severity}
                    </span>
                  </td>

                  <td>
                    <span
                      className={`status-pill ${
                        incident.status === "Reviewed"
                          ? "green-pill"
                          : "amber-pill"
                      }`}
                    >
                      {incident.status}
                    </span>
                  </td>

                  <td>
                    <button
                      className="text-button"
                      onClick={() => onSelect(incident)}
                    >
                      Review →
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="6" className="empty-state">
                  No matching sample incidents found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="table-footer">
        Showing {incidents.length} sample record
        {incidents.length === 1 ? "" : "s"}
      </div>
    </section>
  );
}

/* INCIDENT DETAIL ROW */
function Detail({ label, value }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
