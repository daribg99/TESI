#!/usr/bin/env bash
set -euo pipefail

# ================================================
# cli_openpdc.sh — OpenPDC helper
# Subcommands:
#   addpmu           -> inserisce una PMU (Device + Measurement)
#   addoutputstream  -> segnaposto per futuro
#   help             -> mostra l'help
# ================================================

# ---------- Default global options ----------
NS=""                         # --ns <namespace>  (OBBLIGATORIO)
CLUSTER_PREFIX="cluster1"     # --cluster-prefix
DB_NAME="lower"             # --db
PXC_POD=""                    # --pxc-pod (default: <cluster_prefix>-pxc-0)
HAPROXY_SVC=""                # --haproxy-svc (default: <cluster_prefix>-haproxy)
ROOT_SECRET_NAME=""           # --secret-name (default: <cluster_prefix>-secrets)

# ---------- Helpers ----------
die() { echo "ERRORE: $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }
ok()   { echo "[OK] $*"; }

usage_global() {
  cat <<USAGE
Usage:
  $0 <command> [global options] <subcommand> [command options]

Subcommands:
  addpmu            Add a PMU (Device + Measurements)
  addoutputstream   Create an output stream
  createhistorian   Create a local historian
  help              Show this help

Global options:
  --ns namespace                    mandatory
  --cluster-prefix NAME              (default: cluster1)
  --db NAME                          (default: lower)
  --pxc-pod NAME                     (default: <cluster>-pxc-0)
  --haproxy-svc NAME                 (default: <cluster>-haproxy)
  --secret-name NAME                 (default: <cluster>-secrets)
  --mysql-user USER                  (default: root)

Examples:
  $0 addpmu --ns demo --acronym PMU-3 --name "Pmu-3" --server pmu-3
  $0 addoutputstream   --ns lower   --acronym LOWER   --name low2high   --pmus "PMU-2,PMU-3"   --port 4712 --fps 30 --nomfreq 60
  $0 createhistorian --db higher --ns higher
USAGE
}

# parse only global opts until troviamo il subcommand
GLOBAL_ARGS=()
SUBCOMMAND=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    addpmu|addoutputstream|createhistorian|help) SUBCOMMAND="$1"; shift; break;;
    --ns) NS="$2"; shift 2;;
    --cluster-prefix) CLUSTER_PREFIX="$2"; shift 2;;
    --db) DB_NAME="$2"; shift 2;;
    --pxc-pod) PXC_POD="$2"; shift 2;;
    --haproxy-svc) HAPROXY_SVC="$2"; shift 2;;
    --secret-name) ROOT_SECRET_NAME="$2"; shift 2;;
    -h|--help) usage_global; exit 0;;
    *) GLOBAL_ARGS+=("$1"); shift;;
  esac
done

[[ -z "${SUBCOMMAND}" ]] && { usage_global; exit 1; }

# defaults derivati
[[ -z "$PXC_POD" ]] && PXC_POD="${CLUSTER_PREFIX}-pxc-0"
[[ -z "$HAPROXY_SVC" ]] && HAPROXY_SVC="${CLUSTER_PREFIX}-haproxy"
[[ -z "$ROOT_SECRET_NAME" ]] && ROOT_SECRET_NAME="${CLUSTER_PREFIX}-secrets"

# ======================================================
# SUBCOMMAND: addpmu
# ======================================================
addpmu_usage() {
  cat <<USAGE
Usage:
  $0 addpmu --ns <namespace> --acronym <PMU-ACR> --name <Nome> --server <host/ip> [opzioni] [opzioni globali]

Addpmu options:
  --name NAME            (es. "Pmu-3") OBBLIGATORIO
  --acronym ACR          (es. PMU-3)  
  --server HOST          (endpoint PMU) 
  --port N               (default: 4712)
  --fps N                (default: 25)
  --is-listener true|false   (default: false)
  --hist-id N            (default: 1)
  --user TAG             (default: polito)
  --lon VAL              (default: -98.6)
  --lat VAL              (default: 37.5)

Example:
  $0 addpmu --ns demo --acronym PMU-3 --name "Pmu-3" --server pmu-3 --port 4712 --fps 25
USAGE
}

addoutputstream_usage() {
  cat <<USAGE
Usage:
  $0 addoutputstream --ns <namespace> --acronym <OS-ACR> --name <Nome> [opzioni] [opzioni globali]

Addoutputstream options:
  --acronym ACR              (es. LOWER)                 OBBLIGATORIO
  --name NAME                (es. "low2high")            OBBLIGATORIO
  --pmus "PMU-1,PMU-2,..."   elenco PMU da collegare     OBBLIGATORIO
  --port N                   CommandChannel port (4712 default)
  --fps N                    FramesPerSecond (30 default)
  --nomfreq N                NominalFrequency (60 default)
  --lag MS                   LagTime in ms (3 default)
  --lead MS                  LeadTime in ms (1 default)
  --user TAG                 UpdatedBy/CreatedBy (polito default)

Example:
  $0 addoutputstream --ns lower --acronym LOWER --name low2high --pmus "PMU-3"
USAGE
}

createHistorian_usage(){
  cat <<USAGE
Usage:
  $0 createHistorian --ns <namespace> [opzioni] [opzioni globali]

createhistorian options:
  --name NAME                (default: localhistorian)
  --acronym ACR              (default: LOCAL)
  --connstr STR              ConnectionString (default: NULL)
  --mri N                    MeasurementReportingInterval (default: 100000)
  --user TAG                 UpdatedBy/CreatedBy (default: polito)

Esempi:
  $0 createHistorian --db higher --ns higher 
  $0 createHistorian --ns lower --name myhist --acronym MYHIST --connstr "some=cfg;value=1"
USAGE
}
addpmu_cmd() {
  # default
  NS="default"
  CLUSTER_PREFIX="cluster1"
  DB_NAME="lower"
  POD="${CLUSTER_PREFIX}-pxc-0"
  SVC="${CLUSTER_PREFIX}-haproxy"
  SECRET="${CLUSTER_PREFIX}-secrets"
  FPS=25
  PORT=4712
  NAME=""
  ACRONYM=""
  SERVER=""
  

  # parse args
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
      addpmu_usage
      return 0
      ;;
      --ns) NS="$2"; shift 2;;
      --cluster) CLUSTER_PREFIX="$2"; POD="${CLUSTER_PREFIX}-pxc-0"; SVC="${CLUSTER_PREFIX}-haproxy"; SECRET="${CLUSTER_PREFIX}-secrets"; shift 2;;
      --db) DB_NAME="$2"; shift 2;;
      --name) NAME="$2"; shift 2;;
      --acronym) ACRONYM="$2"; shift 2;;
      --server) SERVER="$2"; shift 2;;
      --fps) FPS="$2"; shift 2;;
      --port) PORT="$2"; shift 2;;
      *) echo "Argument unknown: $1"; return 1;;
    esac
  done

  # ---- derive ACRONYM and SERVER from NAME if missing ----
  if [[ -z "$NAME" && -z "$ACRONYM" ]]; then
    echo "Error: you must specify at least --name or --acronym."
    return 2
  fi

  if [[ -z "$ACRONYM" && -n "$NAME" ]]; then
    ACRONYM="$(echo "$NAME" | tr '[:lower:]' '[:upper:]')"
    ACRONYM="${ACRONYM// /-}"
    ACRONYM="${ACRONYM//_/-}"
    ACRONYM="$(echo "$ACRONYM" | sed 's/[^A-Z0-9-]//g')"
  fi

  if [[ -z "$SERVER" && -n "$NAME" ]]; then
    SERVER="$(echo "$NAME" | tr '[:upper:]' '[:lower:]')"
    SERVER="${SERVER// /-}"
    SERVER="${SERVER//_/-}"
    SERVER="$(echo "$SERVER" | sed 's/[^a-z0-9.-]//g; s/-\{2,\}/-/g; s/^-//; s/-$//')"
  fi

  # If still missing SERVER, use ACRONYM lowercased: es. PMU-3 -> pmu-3
  if [[ -z "$NAME" && -n "$ACRONYM" ]]; then
    # es: PMU-3 -> Pmu-3
    NAME="$(echo "$ACRONYM" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
  fi

  echo "Namespace: $NS"
  echo "DB: $DB_NAME  Pod: $POD  Svc: $SVC"
  echo "Name: $NAME  Acronym: $ACRONYM  Server: $SERVER  FPS: $FPS  Port: $PORT"

  # take ROOT password from k8s secret
  ROOTPWD="$(kubectl get secrets "$SECRET" -n "$NS" -o jsonpath='{.data.root}' | base64 --decode)"

  # exit if acronym already exists  
  EXISTS=$(kubectl exec -n "$NS" -it "$POD" -c pxc -- \
    mysql -h "$SVC" -uroot -p"$ROOTPWD" -N -e "SELECT COUNT(*) FROM ${DB_NAME}.Device WHERE Acronym='${ACRONYM//\'/\'\'}';" 2>/dev/null)
  if [[ "$EXISTS" =~ ^[0-9]+$ ]] && [[ "$EXISTS" -gt 0 && $FORCE -ne 1 ]]; then
    echo "Device acronym '$ACRONYM' already exists in DB '$DB_NAME'. "
    return 3
  fi

  
SQL=$(cat <<EOF
USE \`${DB_NAME}\`;
SET @NodeID := (SELECT ID FROM Node LIMIT 1);
SET @UniqueID := UUID();
SET @AccessID := (SELECT COALESCE(MAX(AccessID), 0) + 1 FROM Device);

INSERT INTO Device (
  NodeID, ParentID, UniqueID, Acronym, Name, IsConcentrator, CompanyID, HistorianID,
  AccessID, VendorDeviceID, ProtocolID, Longitude, Latitude, InterconnectionID,
  ConnectionString, TimeZone, FramesPerSecond, TimeAdjustmentTicks, DataLossInterval,
  ContactList, MeasuredLines, LoadOrder, Enabled, AllowedParsingExceptions,
  ParsingExceptionWindow, DelayedConnectionInterval, AllowUseOfCachedConfiguration,
  AutoStartDataParsingSequence, SkipDisableRealTimeData, MeasurementReportingInterval,
  ConnectOndemand, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn
) VALUES (
  @NodeID, NULL, @UniqueID,
  '${ACRONYM}', '${NAME}', 0, NULL, 1,
  @AccessID, NULL, 1, -98.6, 37.5, NULL,
  'port=${PORT}; maxSendQueueSize=-1; server=${SERVER}; islistener=false; transportprotocol=tcp; interface=0.0.0.0',
  NULL, ${FPS}, 0, 5,
  NULL, NULL, 0, 1, 10, 5, 5, 1, 1, 0,
  100000, 0, 'polito', NOW(6), 'polito', NOW(6)
);

SET @DeviceID := LAST_INSERT_ID();

-- Misure base (freq, df/dt, status)
INSERT INTO Measurement
  (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex,
   SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled,
   UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES
  (1, @DeviceID, '_${ACRONYM}:F',  NULL, 5,  NULL, '${ACRONYM}-FQ', 0, 1, 0, 1, '${NAME} Frequency',                         1, 'polito', NOW(6), 'polito', NOW(6)),
  (1, @DeviceID, '_${ACRONYM}:DF', NULL, 6,  NULL, '${ACRONYM}-DF', 0, 1, 0, 1, '${NAME} Frequency Delta (dF/dt)',          1, 'polito', NOW(6), 'polito', NOW(6)),
  (1, @DeviceID, '_${ACRONYM}:S',  NULL, 8,  NULL, '${ACRONYM}-SF', 0, 1, 0, 1, '${NAME} Status Flags',                      1, 'polito', NOW(6), 'polito', NOW(6));

-- Tensioni A/B/C: magnitudo e angolo
INSERT INTO Measurement
  (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex,
   SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled,
   UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES
  (1, @DeviceID, '_${ACRONYM}-PM1:V',  NULL, 3, 1, '${ACRONYM}-PM1', 0, 1, 0, 1, '${NAME} A  + Voltage Magnitude', 1, 'polito', NOW(6), 'polito', NOW(6)),
  (1, @DeviceID, '_${ACRONYM}-PA1:VH', NULL, 4, 1, '${ACRONYM}-PA1', 0, 1, 0, 1, '${NAME} A  + Voltage Phase Angle', 1, 'polito', NOW(6), 'polito', NOW(6)),
  (1, @DeviceID, '_${ACRONYM}-PM2:V',  NULL, 3, 2, '${ACRONYM}-PM2', 0, 1, 0, 1, '${NAME} B  + Voltage Magnitude', 1, 'polito', NOW(6), 'polito', NOW(6)),
  (1, @DeviceID, '_${ACRONYM}-PA2:VH', NULL, 4, 2, '${ACRONYM}-PA2', 0, 1, 0, 1, '${NAME} B  + Voltage Phase Angle', 1, 'polito', NOW(6), 'polito', NOW(6)),
  (1, @DeviceID, '_${ACRONYM}-PM3:V',  NULL, 3, 3, '${ACRONYM}-PM3', 0, 1, 0, 1, '${NAME} C  + Voltage Magnitude', 1, 'polito', NOW(6), 'polito', NOW(6)),
  (1, @DeviceID, '_${ACRONYM}-PA3:VH', NULL, 4, 3, '${ACRONYM}-PA3', 0, 1, 0, 1, '${NAME} C  + Voltage Phase Angle', 1, 'polito', NOW(6), 'polito', NOW(6));

-- Time Quality Flags
INSERT INTO Measurement
  (HistorianID, DeviceID, PointTag, SignalTypeID, PhasorSourceIndex, SignalReference, Description, Enabled)
VALUES
  (1, @DeviceID, 'GPA_${ACRONYM}:QF', 13, NULL, '${ACRONYM}-QF', '${NAME} Time Quality Flags', 1);
INSERT INTO TrackedChange (TableName, PrimaryKeyColumn, PrimaryKeyValue) VALUES ('Device', 'ID', @DeviceID);
EOF
)

# Stampa la SQL sempre
#echo "---------- BEGIN SQL ----------"
#printf "%s\n" "$SQL"
#echo "----------- END SQL -----------"

printf "%s" "$SQL" | kubectl exec -i "$POD" -c pxc -n "$NS" -- \
  mysql -h "$SVC" -uroot -p"$ROOTPWD" --database "$DB_NAME" --batch --silent

echo "PMU '$NAME' ($ACRONYM) successfully added on db $DB_NAME."
}


# ======================================================
# SUBCOMMAND: addoutputstream (placeholder)
# ======================================================
addoutputstream_cmd() {
  # defaults
  local ACRONYM="" NAME="" PMUS="" PORT=4712 FPS=30 NOMFREQ=60 LAG=3 LEAD=1 USERTAG="polito"
  local NS_LOCAL="$NS" POD="$PXC_POD" SVC="$HAPROXY_SVC" DB="$DB_NAME" SECRET="$ROOT_SECRET_NAME"

  # parse args
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
      addoutputstream_usage
      return 0
      ;;
      --acronym) ACRONYM="$2"; shift 2;;
      --name) NAME="$2"; shift 2;;
      --pmus) PMUS="$2"; shift 2;;
      --port) PORT="$2"; shift 2;;
      --fps) FPS="$2"; shift 2;;
      --nomfreq) NOMFREQ="$2"; shift 2;;
      --lag) LAG="$2"; shift 2;;
      --lead) LEAD="$2"; shift 2;;
      --user) USERTAG="$2"; shift 2;;
      *) echo "Argument unknown: $1"; return 1;;
    esac
  done

  [[ -z "$NS_LOCAL" || -z "$ACRONYM" || -z "$NAME" || -z "$PMUS" ]] && { addoutputstream_usage; return 2; }

  echo "Namespace: $NS_LOCAL"
  echo "DB: $DB  Pod: $POD  Svc: $SVC"
  echo "OutputStream: $NAME ($ACRONYM)  PMUs: $PMUS  FPS: $FPS  Port: $PORT"

  # root pwd da secret
  ROOTPWD="$(kubectl get secrets "$SECRET" -n "$NS_LOCAL" -o jsonpath='{.data.root}' | base64 --decode)"

  # prepara SQL base per OutputStream
  SQL=$(cat <<EOF
USE \`${DB}\`;
SET @NodeID := (SELECT ID FROM Node LIMIT 1);

INSERT INTO OutputStream (
  NodeID, Acronym, Name, Type, ConnectionString, IDCode, CommandChannel, DataChannel,
  AutoPublishConfigFrame, AutoStartDataChannel, NominalFrequency, FramesPerSecond,
  LagTime, LeadTime, UseLocalClockAsRealTime, AllowSortsByArrival, LoadOrder, Enabled,
  IgnoreBadTimeStamps, TimeResolution, AllowPreemptivePublishing, DownSamplingMethod,
  DataFormat, CoordinateFormat, CurrentScalingValue, VoltageScalingValue, AnalogScalingValue,
  DigitalMaskValue, PerformTimeReasonabilityCheck, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn
) VALUES (
  @NodeID, '${ACRONYM}', '${NAME}', 0, NULL, 0,
  CONCAT('port=${PORT};'), NULL,
  0, 1, ${NOMFREQ}, ${FPS},
  ${LAG}, ${LEAD}, 1, 0, 1, 1,
  0, 330000, 1, 'LastReceived', 'FloatingPoint', 'Polar',
  2423, 2725785, 1373291, -65536, 1,
  '${USERTAG}', NOW(6), '${USERTAG}', NOW(6)
);

SET @AdapterID := LAST_INSERT_ID();
EOF
  )
  # for each PMU, add Device, Phasors, Analogs, Measurements
  IFS=',' read -ra PMU_ARR <<< "$PMUS"
  idcode=1
  for raw in "${PMU_ARR[@]}"; do
    pmu="$(echo "$raw" | xargs)"         # trim
    [[ -z "$pmu" ]] && continue
    pmu_name="$(echo "$pmu" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"

    BLOCK=$(cat <<EOF
-- ======================================================
-- PMU ${pmu}
-- Device dello stream
INSERT INTO OutputStreamDevice (
  NodeID, AdapterID, IDCode, Acronym, BpaAcronym, Name,
  PhasorDataFormat, FrequencyDataFormat, AnalogDataFormat, CoordinateFormat,
  LoadOrder, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn
) VALUES (
  @NodeID, @AdapterID, ${idcode}, '${pmu}', '', '${pmu_name}',
  NULL, NULL, NULL, NULL,
  0, 1, '${USERTAG}', NOW(6), '${USERTAG}', NOW(6)
);
SET @OSDevID := LAST_INSERT_ID();

-- Phasor A/B/C (tensioni)
INSERT INTO OutputStreamDevicePhasor
  (NodeID, OutputStreamDeviceID, Label, Type, Phase, ScalingValue, LoadOrder, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES
  (@NodeID, @OSDevID, 'A', 'V', '+', 0, 1, '${USERTAG}', NOW(6), '${USERTAG}', NOW(6)),
  (@NodeID, @OSDevID, 'B', 'V', '+', 0, 2, '${USERTAG}', NOW(6), '${USERTAG}', NOW(6)),
  (@NodeID, @OSDevID, 'C', 'V', '+', 0, 3, '${USERTAG}', NOW(6), '${USERTAG}', NOW(6));


INSERT INTO OutputStreamDeviceAnalog
  (NodeID, OutputStreamDeviceID, Label, Type, ScalingValue, LoadOrder, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES
  (@NodeID, @OSDevID, 'D', 0, 0, 1, '${USERTAG}', NOW(6), '${USERTAG}', NOW(6));

INSERT INTO OutputStreamMeasurement
  (NodeID, AdapterID, HistorianID, PointID, SignalReference, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
SELECT
  @NodeID, @AdapterID, 1, m.PointID, m.SignalReference, '${USERTAG}', NOW(6), '${USERTAG}', NOW(6)
FROM Measurement m
JOIN Device d ON d.ID = m.DeviceID
WHERE d.Acronym='${pmu}'
  AND m.SignalReference IN (
    '${pmu}-AV1','${pmu}-FQ','${pmu}-DF','${pmu}-SF',
    '${pmu}-PM1','${pmu}-PA1',
    '${pmu}-PM2','${pmu}-PA2',
    '${pmu}-PM3','${pmu}-PA3'
  );
EOF
)
SQL+="$BLOCK"
    idcode=$((idcode+1))
  done


  # esegui
   printf "%s" "$SQL" | kubectl exec -i "$POD" -c pxc -n "$NS_LOCAL" -- \
   mysql -h "$SVC" -uroot -p"$ROOTPWD" --database "$DB" --batch --silent

   # Stampa SQL per debug
    #echo "---------- BEGIN SQL ----------"
    #printf "%s\n" "$SQL"
    #echo "----------- END SQL -----------"

  ok "OutputStream '${NAME}' (${ACRONYM}) successfully created with PMUs: ${PMUS}"
}

createhistorian_cmd() {
 
  local USERTAG="polito"

  # historian fields
  local NAME="localhistorian"
  local ACRONYM="LOCAL"
  local CONNSTR=""    # se vuoto -> NULL
  local MRI=100000    # MeasurementReportingInterval

  if [[ $# -eq 0 ]]; then
    createHistorian_usage
    return 0
  fi

  # --- parsing ---
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) createHistorian_usage; return 0;;
      --name) NAME="$2"; shift 2;;
      --acronym) ACRONYM="$2"; shift 2;;
      --connstr) CONNSTR="$2"; shift 2;;
      --mri) MRI="$2"; shift 2;;
      *) echo "Argument unknown: $1"; return 1;;
    esac
  done

  echo "Namespace: $NS_LOCAL"
  echo "DB: $DB_NAME  Pod: $POD  Svc: $SVC"
  echo "Historian: $NAME ($ACRONYM)"

  # --- root pwd dal secret ---
  local ROOTPWD
  ROOTPWD="$(kubectl get secrets "$SECRET" -n "$NS_LOCAL" -o jsonpath='{.data.root}' | base64 --decode)"

  # --- costruzione SQL ---
  # Se CONNSTR è vuoto, usa NULL; altrimenti quota la stringa
  local CONN_SQL="NULL"
  if [[ -n "$CONNSTR" ]]; then
    # escape apici singoli
    local ESCONN
    ESCONN="$(printf "%s" "$CONNSTR" | sed "s/'/''/g")"
    CONN_SQL="'${ESCONN}'"
  fi

  SQL=$(cat <<EOF
USE \`${DB_NAME}\`;


SET @NodeID := (SELECT ID FROM Node LIMIT 1);


INSERT INTO Historian (
  NodeID, Acronym, Name, AssemblyName, TypeName, ConnectionString,
  IsLocal, MeasurementReportingInterval, Description, LoadOrder, Enabled,
  UpdatedBy, UpdatedOn, CreatedBy, CreatedOn
) VALUES (
  @NodeID,
  '${ACRONYM}',
  '${NAME}',
  'HistorianAdapters.dll',
  'HistorianAdapters.LocalOutputAdapter',
  ${CONN_SQL},
  1,
  ${MRI},
  NULL,
  0,
  1,
  '${USERTAG}', NOW(6), '${USERTAG}', NOW(6)
);

EOF
)

  
    #echo "---------- BEGIN SQL ----------"
    #printf "%s\n" "$SQL"
    #echo "----------- END SQL -----------"
  

  printf "%s" "$SQL" | kubectl exec -i "$POD" -c pxc -n "$NS_LOCAL" -- \
    mysql -h "$SVC" -uroot -p"$ROOTPWD" --database "$DB_NAME" --batch --silent

  echo "[OK] Historian '${NAME}' (${ACRONYM}) successfully created."
}
# ======================================================
# DISPATCH
# ======================================================
case "$SUBCOMMAND" in
  help) usage_global;;
  addpmu) addpmu_cmd "$@";;
  addoutputstream) addoutputstream_cmd "${GLOBAL_ARGS[@]}" "$@";;
  createhistorian) createhistorian_cmd "${GLOBAL_ARGS[@]}" "$@";;
  *) usage_global; exit 1;;
esac
