
import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8001";

// --------------------------------------------------
// FORMAT INCIDENT RECORDS
// --------------------------------------------------

function formatIncident(row) {
  const date = row.detected_at
    ? new Date(row.detected_at)
    : null;

  let time = "—";

  if (date && !Number.isNaN(date.getTime())) {
    time = date.toLocaleTimeString();
  } else if (row.video_time_seconds != null) {
    const seconds = Math.floor(Number(row.video_time_seconds));
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;

    time = `${String(minutes).padStart(2, "0")}:${String(
      remainingSeconds
    ).padStart(2, "0")}`;
  }

  const status =
    String(row.status || "Needs review").toLowerCase() === "reviewed"
      ? "Reviewed"
      : "Needs review";

  return {
    dbId: row.incident_id,
    id: `VA-${String(row.incident_id).padStart(3, "0")}`,
    type: row.incident_type || "Possible accident",
    camera: row.camera_id || "Unknown camera",
    time,
    severity: row.severity || "Unclassified",
    status,
    location: row.location || "Location unavailable",
    confidence: row.confidence,
    evidence: row.evidence,
    videoPath: row.video_path,
    videoTime: row.video_time_seconds,
  };
}

// --------------------------------------------------
// SIDEBAR NAVIGATION
// --------------------------------------------------

const navigation = [
  { icon: "▦", label: "Overview" },
  { icon: "▣", label: "CCTV Monitoring" },
  { icon: "⚠", label: "Incidents" },
  { icon: "⌖", label: "Locations" },
  { icon: "♧", label: "Emergency Alerts" },
  { icon: "☷", label: "Reports" },
];

// --------------------------------------------------
// MAIN APPLICATION
// --------------------------------------------------

function App() {
  const [incidents, setIncidents] = useState([]);
  const [notifications, setNotifications] = useState([]);

  const [activeNav, setActiveNav] = useState("Overview");
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null);

  const [loading, setLoading] = useState(true);
  const [notificationLoading, setNotificationLoading] = useState(false);

  const [apiError, setApiError] = useState("");
  const [notificationError, setNotificationError] = useState("");
  const [notificationMessage, setNotificationMessage] = useState("");

  const [refreshKey, setRefreshKey] = useState(0);
  const [updating, setUpdating] = useState(false);
  const [notifying, setNotifying] = useState(false);

  // --------------------------------------------------
  // FETCH INCIDENTS FROM FASTAPI / MYSQL
  // --------------------------------------------------

  async function fetchIncidents() {
    setLoading(true);
    setApiError("");

    try {
      const response = await fetch(`${API_BASE_URL}/incidents`);

      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }

      const data = await response.json();

      const records = Array.isArray(data)
        ? data
        : data.incidents || [];

      const formattedRecords = records
        .map(formatIncident)
        .sort((a, b) => b.dbId - a.dbId);

      setIncidents(formattedRecords);
    } catch (error) {
      console.error("Incident fetching error:", error);

      setApiError(
        "Unable to connect to the Vita Aegis backend. Please check whether FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  }

  // --------------------------------------------------
  // FETCH EMERGENCY NOTIFICATIONS
  // --------------------------------------------------

  async function fetchNotifications() {
    setNotificationLoading(true);
    setNotificationError("");

    try {
      const response = await fetch(`${API_BASE_URL}/notifications`);

      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }

      const data = await response.json();

      setNotifications(data.notifications || []);
    } catch (error) {
      console.error("Notification fetching error:", error);

      setNotificationError(
        "Unable to load emergency notifications. Please check the backend."
      );
    } finally {
      setNotificationLoading(false);
    }
  }

  // --------------------------------------------------
  // LOAD DATABASE RECORDS
  // --------------------------------------------------

  useEffect(() => {
    fetchIncidents();
    fetchNotifications();
  }, [refreshKey]);

  // --------------------------------------------------
  // SEARCH AND FILTER INCIDENTS
  // --------------------------------------------------

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

      return matchesFilter && text.includes(search.toLowerCase());
    });
  }, [incidents, filter, search]);

  const pendingCount = incidents.filter(
    (item) => item.status === "Needs review"
  ).length;

  const reviewedCount = incidents.filter(
    (item) => item.status === "Reviewed"
  ).length;

  const policeCount = notifications.filter(
    (item) => item.service_type === "POLICE"
  ).length;

  const hospitalCount = notifications.filter(
    (item) => item.service_type === "HOSPITAL"
  ).length;

  // --------------------------------------------------
  // MARK INCIDENT AS REVIEWED
  // --------------------------------------------------

  async function markReviewed(incident) {
    if (!incident?.dbId || updating) return;

    setUpdating(true);
    setApiError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/incidents/${incident.dbId}/status`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            status: "Reviewed",
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Unable to update incident. HTTP ${response.status}`
        );
      }

      setIncidents((items) =>
        items.map((item) =>
          item.dbId === incident.dbId
            ? { ...item, status: "Reviewed" }
            : item
        )
      );

      setSelected((current) =>
        current?.dbId === incident.dbId
          ? { ...current, status: "Reviewed" }
          : current
      );
    } catch (error) {
      console.error("Review update error:", error);

      setApiError(
        "Could not update the incident status. Please try again."
      );
    } finally {
      setUpdating(false);
    }
  }

  // --------------------------------------------------
  // GENERATE DEMO EMERGENCY NOTIFICATIONS
  // --------------------------------------------------

  async function generateDemoAlerts(incident) {
    if (!incident?.dbId || notifying) return;

    setNotifying(true);
    setNotificationError("");
    setNotificationMessage("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/incidents/${incident.dbId}/notify`,
        {
          method: "POST",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Unable to generate demo notifications."
        );
      }

      const count = data.notifications?.length || 0;

      setNotificationMessage(
        `Successfully processed ${count} demo notification(s) for ${incident.id}.`
      );

      // Refresh notification records from MySQL
      await fetchNotifications();

      // Open Emergency Alerts page after generation
      setActiveNav("Emergency Alerts");
    } catch (error) {
      console.error("Emergency notification error:", error);

      setNotificationError(error.message);
    } finally {
      setNotifying(false);
    }
  }

  // --------------------------------------------------
  // REFRESH ALL DATABASE DATA
  // --------------------------------------------------

  function refreshDashboard() {
    setRefreshKey((value) => value + 1);
  }

  // --------------------------------------------------
  // MAIN UI
  // --------------------------------------------------

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

              {item.label === "Incidents" && pendingCount > 0 && (
                <span className="nav-count">
                  {pendingCount}
                </span>
              )}

              {item.label === "Emergency Alerts" &&
                notifications.length > 0 && (
                  <span className="nav-count">
                    {notifications.length}
                  </span>
                )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div
            className={`system-dot ${
              apiError ? "offline-dot" : ""
            }`}
          />

          <div>
            <strong>
              {apiError
                ? "Backend disconnected"
                : "Prototype running"}
            </strong>

            <span>Local demonstration</span>
          </div>
        </div>

      </aside>

      {/* MAIN CONTENT */}

      <main className="main-content">

        {/* HEADER */}

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
          <strong>PROTOTYPE</strong>

          <span>
            Incident records and simulated emergency notifications
            are retrieved from the local MySQL database through
            FastAPI. Recorded footage is used for demonstration.
            NO REAL EMERGENCY ALERTS ARE SENT.
          </span>
        </div>

        {/* API ERROR */}

        {apiError && (
          <div className="notice">
            <strong>CONNECTION ERROR</strong>

            <span>{apiError}</span>

            <button
              className="text-button"
              onClick={refreshDashboard}
            >
              Retry
            </button>
          </div>
        )}

        {/* ==================================================
            OVERVIEW PAGE
        ================================================== */}

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
                label="Total incidents"
                value={String(incidents.length).padStart(2, "0")}
                detail="Database records"
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
                label="Reviewed incidents"
                value={String(reviewedCount).padStart(2, "0")}
                detail="Review completed"
                icon="✓"
                tone="green"
              />

            </section>

            {/* CCTV AND SYSTEM STATUS */}

            <section className="content-grid">

              {/* CCTV PANEL */}

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

                  <div className="camera-video">

                    <video
                      controls
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

                  <span>Camera 01 / 04</span>

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

                  <span className="status-symbol">▣</span>

                  <div className="status-copy">
                    <strong>Database connection</strong>

                    <span>
                      {apiError
                        ? "Backend unavailable"
                        : loading
                        ? "Checking connection..."
                        : "Incident records retrieved"}
                    </span>
                  </div>

                  <span
                    className={`status-pill ${
                      apiError || loading
                        ? "amber-pill"
                        : "green-pill"
                    }`}
                  >
                    {apiError
                      ? "Offline"
                      : loading
                      ? "Checking"
                      : "Connected"}
                  </span>

                </div>

                <div className="status-row">

                  <span className="status-symbol">⌁</span>

                  <div className="status-copy">
                    <strong>AI detection</strong>
                    <span>Python analysis prototype</span>
                  </div>

                  <span className="status-pill amber-pill">
                    Prototype
                  </span>

                </div>

                <div className="status-row">

                  <span className="status-symbol">♧</span>

                  <div className="status-copy">
                    <strong>Emergency alerts</strong>
                    <span>
                      Demo notification system connected
                    </span>
                  </div>

                  <span className="status-pill green-pill">
                    Demo Ready
                  </span>

                </div>

                <div className="info-box">
                  <strong>Emergency notification status</strong>

                  <p>
                    {notifications.length} simulated notification(s)
                    saved in the database. No real alerts are sent.
                  </p>

                  <button
                    className="text-button"
                    onClick={() => setActiveNav("Emergency Alerts")}
                  >
                    View emergency alerts →
                  </button>
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
              loading={loading}
              onRefresh={refreshDashboard}
            />

          </>

        ) : activeNav === "Incidents" ? (

          /* ==================================================
             INCIDENTS PAGE
          ================================================== */

          <IncidentTable
            incidents={filteredIncidents}
            filter={filter}
            setFilter={setFilter}
            search={search}
            setSearch={setSearch}
            onSelect={setSelected}
            loading={loading}
            onRefresh={refreshDashboard}
          />

        ) : activeNav === "CCTV Monitoring" ? (

          /* ==================================================
             CCTV MONITORING PAGE
          ================================================== */

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

        ) : activeNav === "Emergency Alerts" ? (

          /* ==================================================
             EMERGENCY ALERTS PAGE
          ================================================== */

          <EmergencyAlertsPage
            notifications={notifications}
            policeCount={policeCount}
            hospitalCount={hospitalCount}
            loading={notificationLoading}
            error={notificationError}
            message={notificationMessage}
            onRefresh={fetchNotifications}
          />

        ) : activeNav === "Locations" ? (

          /* ==================================================
             LOCATIONS PAGE
          ================================================== */

          <section className="panel placeholder-page">

            <div className="camera-symbol">⌖</div>

            <h2>Highway Locations</h2>

            <p>
              Camera location details will be integrated
              in this section.
            </p>

            <button
              className="primary-button"
              onClick={() => setActiveNav("Overview")}
            >
              Back to overview
            </button>

          </section>

        ) : (

          /* ==================================================
             REPORTS PAGE
          ================================================== */

          <section className="panel placeholder-page">

            <div className="camera-symbol">☷</div>

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

        {/* FOOTER */}

        <footer>
          VITA AEGIS · Road Accident Monitoring Prototype

          <span>For demonstration only</span>
        </footer>

      </main>

      {/* ==================================================
          INCIDENT DETAILS MODAL
      ================================================== */}

      {selected && (

        <div
          className="modal-backdrop"
          onClick={() => {
            if (!updating && !notifying) {
              setSelected(null);
            }
          }}
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
                disabled={updating || notifying}
                onClick={() => setSelected(null)}
              >
                ×
              </button>

            </div>

            <div className="notice">

              <strong>PROTOTYPE</strong>

              <span>
                This record is retrieved from the local
                incident database. It requires human review
                before being treated as a confirmed accident.
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
                label="Confidence"
                value={
                  selected.confidence != null
                    ? `${(
                        Number(selected.confidence) * 100
                      ).toFixed(1)}%`
                    : "Not available"
                }
              />

              <Detail
                label="Review status"
                value={selected.status}
              />

              {selected.videoTime != null && (
                <Detail
                  label="Video timestamp"
                  value={`${selected.videoTime} seconds`}
                />
              )}

            </div>

            {/* GENERATE DEMO EMERGENCY ALERTS */}

            <button
              className="primary-button full-button"
              disabled={notifying}
              onClick={() => generateDemoAlerts(selected)}
            >
              {notifying
                ? "Generating demo alerts..."
                : "Generate Demo Emergency Alerts"}
            </button>

            {notificationMessage && (
              <div className="info-box">
                {notificationMessage}
              </div>
            )}

            {notificationError && (
              <div className="notice">
                <strong>NOTIFICATION ERROR</strong>
                <span>{notificationError}</span>
              </div>
            )}

            {/* MARK AS REVIEWED */}

            {selected.status !== "Reviewed" && (

              <button
                className="primary-button full-button"
                disabled={updating}
                onClick={() => markReviewed(selected)}
              >
                {updating
                  ? "Updating..."
                  : "Mark as reviewed"}
              </button>

            )}

            {selected.status === "Reviewed" && (
              <div className="info-box">
                This incident has been reviewed.
              </div>
            )}

          </section>

        </div>

      )}

    </div>
  );
}

// --------------------------------------------------
// STATISTICS CARD
// --------------------------------------------------

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

// --------------------------------------------------
// INCIDENT TABLE
// --------------------------------------------------

function IncidentTable({
  incidents,
  filter,
  setFilter,
  search,
  setSearch,
  onSelect,
  loading,
  onRefresh,
}) {
  return (
    <section className="panel incidents-panel">

      <div className="panel-heading incident-heading">

        <div>
          <h3>Incident Records</h3>
          <p>Records retrieved from the MySQL database</p>
        </div>

        <div className="table-controls">

          <input
            aria-label="Search incidents"
            placeholder="Search incidents..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />

          <select
            aria-label="Filter by status"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          >
            <option>All</option>
            <option>Needs review</option>
            <option>Reviewed</option>
          </select>

          <button
            className="text-button"
            onClick={onRefresh}
            disabled={loading}
          >
            {loading ? "Loading..." : "↻ Refresh"}
          </button>

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

            {loading ? (

              <tr>
                <td colSpan="6" className="empty-state">
                  Loading incident records...
                </td>
              </tr>

            ) : incidents.length ? (

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
                  No matching database incidents found.
                </td>
              </tr>

            )}

          </tbody>

        </table>

      </div>

      <div className="table-footer">
        Showing {incidents.length} database record
        {incidents.length === 1 ? "" : "s"}
      </div>

    </section>
  );
}

// --------------------------------------------------
// EMERGENCY ALERTS PAGE
// --------------------------------------------------

function EmergencyAlertsPage({
  notifications,
  policeCount,
  hospitalCount,
  loading,
  error,
  message,
  onRefresh,
}) {
  return (
    <section className="panel incidents-panel">

      <div className="panel-heading">

        <div>
          <h3>Emergency Notifications</h3>

          <p>
            Simulated police and hospital alerts from MySQL
          </p>
        </div>

        <button
          className="text-button"
          onClick={onRefresh}
          disabled={loading}
        >
          {loading ? "Loading..." : "↻ Refresh"}
        </button>

      </div>

      {/* DEMO WARNING */}

      <div className="notice">

        <strong>DEMO ONLY</strong>

        <span>
          These notifications are simulated and saved locally.
          NO REAL ALERTS ARE SENT TO POLICE OR HOSPITALS.
        </span>

      </div>

      {/* SUCCESS MESSAGE */}

      {message && (
        <div className="info-box">
          <strong>Notification result</strong>
          <p>{message}</p>
        </div>
      )}

      {/* ERROR MESSAGE */}

      {error && (
        <div className="notice">
          <strong>ERROR</strong>
          <span>{error}</span>
        </div>
      )}

      {/* NOTIFICATION STATISTICS */}

      <section className="stats-grid">

        <StatCard
          label="Total notifications"
          value={String(notifications.length).padStart(2, "0")}
          detail="Saved in database"
          icon="♧"
          tone="blue"
        />

        <StatCard
          label="Police alerts"
          value={String(policeCount).padStart(2, "0")}
          detail="Simulated notifications"
          icon="⚑"
          tone="orange"
        />

        <StatCard
          label="Hospital alerts"
          value={String(hospitalCount).padStart(2, "0")}
          detail="Simulated notifications"
          icon="✚"
          tone="green"
        />

      </section>

      {/* NOTIFICATION TABLE */}

      <div className="table-wrap">

        <table>

          <thead>
            <tr>
              <th>INCIDENT</th>
              <th>SERVICE</th>
              <th>TYPE</th>
              <th>STATUS</th>
              <th>CREATED AT</th>
            </tr>
          </thead>

          <tbody>

            {loading ? (

              <tr>
                <td colSpan="5" className="empty-state">
                  Loading emergency notifications...
                </td>
              </tr>

            ) : notifications.length > 0 ? (

              notifications.map((item) => (

                <tr key={item.notification_id}>

                  <td>
                    <strong>
                      VA-{String(item.incident_id).padStart(3, "0")}
                    </strong>

                    <span className="subcell">
                      {item.incident_type}
                    </span>
                  </td>

                  <td>{item.service_name}</td>

                  <td>{item.service_type}</td>

                  <td>
                    <span className="status-pill green-pill">
                      {item.status}
                    </span>
                  </td>

                  <td>
                    {item.created_at
                      ? new Date(item.created_at).toLocaleString()
                      : "—"}
                  </td>

                </tr>

              ))

            ) : (

              <tr>
                <td colSpan="5" className="empty-state">
                  No emergency notifications found.
                  Generate demo alerts from an incident record.
                </td>
              </tr>

            )}

          </tbody>

        </table>

      </div>

      {/* NOTIFICATION MESSAGE DETAILS */}

      <div className="notification-messages">

        <div className="panel-heading">
          <div>
            <h3>Notification Details</h3>
            <p>Messages saved in the emergency notification database</p>
          </div>
        </div>

        {notifications.map((item) => (

          <div
            className="info-box"
            key={`message-${item.notification_id}`}
          >

            <strong>
              {item.service_name} — {item.notification_type}
            </strong>

            <p>{item.message}</p>

          </div>

        ))}

        {!loading && notifications.length === 0 && (
          <p className="empty-state">
            Notification message details will appear here.
          </p>
        )}

      </div>

      {/* FINAL DEMO DISCLAIMER */}

      <div className="info-box">
        <strong>Simulation status</strong>

        <p>
          All emergency notifications are stored with SIMULATED
          status. This prototype does not contact real police
          stations, hospitals, or emergency services.
        </p>
      </div>

    </section>
  );
}

// --------------------------------------------------
// INCIDENT DETAIL ROW
// --------------------------------------------------

function Detail({ label, value }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
