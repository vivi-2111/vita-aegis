import math
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load database settings from backend/.env
load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(
    title="Vita Aegis API",
    description="Backend API for highway incident monitoring",
    version="1.0.0"
)

# Allow the React development server to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_connection():
    try:
        return mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME", "vita_aegis_db")
        )
    except Error as error:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {error}"
        )


class IncidentCreate(BaseModel):
    incident_type: str
    camera_id: Optional[str] = "Camera 01"
    detected_at: Optional[datetime] = None
    video_time_seconds: Optional[float] = None
    vehicle_ids: Optional[str] = None
    evidence: Optional[str] = None
    severity: Optional[str] = "Unclassified"
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    location: Optional[str] = None
    video_path: Optional[str] = None
    status: Optional[str] = "Needs Review"


class IncidentStatusUpdate(BaseModel):
    status: str


@app.get("/")
def home():
    return {
        "message": "Vita Aegis API is running",
        "database": "vita_aegis_db"
    }


@app.get("/health/db")
def database_health():
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE()")
        database = cursor.fetchone()[0]
        cursor.close()
        return {"database_connected": True, "database": database}
    finally:
        connection.close()


@app.get("/incidents")
def get_incidents():
    connection = get_connection()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM incidents ORDER BY detected_at DESC"
        )
        records = cursor.fetchall()
        cursor.close()
        return records
    finally:
        connection.close()


@app.post("/incidents", status_code=201)
def create_incident(incident: IncidentCreate):
    connection = get_connection()
    try:
        cursor = connection.cursor()

        query = """
            INSERT INTO incidents (
                incident_type, camera_id, detected_at,
                video_time_seconds, vehicle_ids, evidence,
                severity, confidence, location, video_path, status
            )
            VALUES (
                %s, %s, COALESCE(%s, CURRENT_TIMESTAMP),
                %s, %s, %s, %s, %s, %s, %s, %s
            )
        """

        values = (
            incident.incident_type,
            incident.camera_id,
            incident.detected_at,
            incident.video_time_seconds,
            incident.vehicle_ids,
            incident.evidence,
            incident.severity,
            incident.confidence,
            incident.location,
            incident.video_path,
            incident.status
        )

        cursor.execute(query, values)
        connection.commit()
        new_id = cursor.lastrowid
        cursor.close()

        return {
            "message": "Incident saved successfully",
            "incident_id": new_id
        }
    finally:
        connection.close()


@app.patch("/incidents/{incident_id}/status")
def update_incident_status(
    incident_id: int,
    update: IncidentStatusUpdate
):
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE incidents SET status = %s WHERE incident_id = %s",
            (update.status, incident_id)
        )
        connection.commit()

        if cursor.rowcount == 0:
            cursor.close()
            raise HTTPException(
                status_code=404,
                detail="Incident not found"
            )

        cursor.close()
        return {
            "message": "Incident status updated",
            "incident_id": incident_id,
            "status": update.status
        }
    finally:
        connection.close()

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in kilometres."""
    earth_radius = 6371

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return earth_radius * 2 * math.asin(math.sqrt(a))


@app.post("/incidents/{incident_id}/notify")
def notify_emergency_services(incident_id: int):
    connection = get_connection()

    try:
        cursor = connection.cursor(dictionary=True)

        # 1. Get incident
        cursor.execute(
            "SELECT * FROM incidents WHERE incident_id = %s",
            (incident_id,)
        )
        incident = cursor.fetchone()

        if not incident:
            raise HTTPException(
                status_code=404,
                detail="Incident not found"
            )

        # 2. Find the camera location
        cursor.execute(
            "SELECT * FROM cameras WHERE camera_id = %s",
            (incident["camera_id"],)
        )
        camera = cursor.fetchone()

        if not camera or camera["latitude"] is None or camera["longitude"] is None:
            raise HTTPException(
                status_code=400,
                detail="Camera GPS coordinates are not registered"
            )

        camera_lat = float(camera["latitude"])
        camera_lon = float(camera["longitude"])

        # 3. Get demo emergency services
        cursor.execute("""
            SELECT * FROM emergency_services
            WHERE is_demo = TRUE
            AND service_type IN ('POLICE', 'HOSPITAL')
        """)
        services = cursor.fetchall()

        notifications = []

        # 4. Find nearest police station and hospital
        for service_type in ("POLICE", "HOSPITAL"):

            matching_services = [
                service for service in services
                if service["service_type"] == service_type
            ]

            if not matching_services:
                continue

            nearest = min(
                matching_services,
                key=lambda service: calculate_distance(
                    camera_lat,
                    camera_lon,
                    float(service["latitude"]),
                    float(service["longitude"])
                )
            )

            distance = calculate_distance(
                camera_lat,
                camera_lon,
                float(nearest["latitude"]),
                float(nearest["longitude"])
            )

            # 5. Prevent duplicate notifications
            cursor.execute("""
                SELECT notification_id, status
                FROM emergency_notifications
                WHERE incident_id = %s AND service_id = %s
            """, (incident_id, nearest["service_id"]))

            existing = cursor.fetchone()

            if existing:
                notifications.append({
                    "service_name": nearest["service_name"],
                    "service_type": service_type,
                    "distance_km": round(distance, 2),
                    "notification_id": existing["notification_id"],
                    "status": existing["status"],
                    "message": "Notification already exists"
                })
                continue

            # 6. Create clearly labelled demo alert
            message = (
                "DEMO ALERT - NOT SENT TO EMERGENCY SERVICES. "
                f"Incident: {incident['incident_type']}. "
                f"Camera: {incident['camera_id']}. "
                f"Location: {incident['location'] or camera['location_name']}. "
                f"Nearest service: {nearest['service_name']}. "
                f"Incident time: {incident['detected_at']}."
            )

            cursor.execute("""
                INSERT INTO emergency_notifications
                (incident_id, service_id, notification_type, message, status)
                VALUES (%s, %s, %s, %s, 'SIMULATED')
            """, (
                incident_id,
                nearest["service_id"],
                service_type,
                message
            ))

            notifications.append({
                "notification_id": cursor.lastrowid,
                "service_name": nearest["service_name"],
                "service_type": service_type,
                "distance_km": round(distance, 2),
                "status": "SIMULATED",
                "message": message
            })

        connection.commit()

        return {
            "incident_id": incident_id,
            "demo_mode": True,
            "warning": "DEMO ONLY - NO REAL ALERTS SENT",
            "notifications": notifications
        }

    finally:
        connection.close()


@app.get("/notifications")
def get_notifications():
    connection = get_connection()

    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                n.notification_id,
                n.incident_id,
                n.service_id,
                n.notification_type,
                n.message,
                n.status,
                n.created_at,
                s.service_name,
                s.service_type,
                i.incident_type,
                i.camera_id,
                i.location
            FROM emergency_notifications n
            JOIN emergency_services s
                ON n.service_id = s.service_id
            JOIN incidents i
                ON n.incident_id = i.incident_id
            ORDER BY n.created_at DESC
        """)

        records = cursor.fetchall()

        cursor.close()

        return {
            "demo_mode": True,
            "notifications": records
        }

    finally:
        connection.close()
