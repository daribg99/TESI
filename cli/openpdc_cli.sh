#!/usr/bin/env bash
set -euo pipefail

# ---------- Default global options ----------
NS=""                         # --ns <namespace>  (OBBLIGATORIO)
CLUSTER_PREFIX="cluster1"     # --cluster-prefix
DB_NAME=""             # --db
PXC_POD=""                    # --pxc-pod (default: <cluster_prefix>-pxc-0)
HAPROXY_SVC=""                # --haproxy-svc (default: <cluster_prefix>-haproxy)
ROOT_SECRET_NAME=""           # --secret-name (default: <cluster_prefix>-secrets)
OPENPDC_POD=""                  # --pod <podname>  (OBBLIGATORIO)
# ---------- Helpers ----------
die() { echo "ERRORE: $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }
ok()   { echo "[OK] $*"; }

usage_global() {
  cat <<USAGE
Usage:
  $0 <subcommand> [global options] [command options]

Subcommands:
  addpmu            Add a PMU (Device + Measurements)
  createoutputstream   Create an output stream
  createhistorian   Create a local historian
  connectiontopdc   Connect to lower level PDC
  createaccount   Create a MySQL user account
  help              Show this help

Global options:
  --ns namespace                     mandatory
  --db NAME                          mandatory
  --pod NAME                         (es. openpdc-pod-1234-5678) mandatory
  --cluster-prefix NAME              (default: cluster1)
  --pxc-pod NAME                     (default: <cluster>-pxc-0)
  --haproxy-svc NAME                 (default: <cluster>-haproxy)
  --secret-name NAME                 (default: <cluster>-secrets)
  --mysql-user USER                  (default: root)

addpmu options:
  --name NAME            (es. "Pmu-3") mandatory
  --acronym ACR          (es. PMU-3)  
  --server HOST          (endpoint PMU) 
  --port N               (default: 4712)
  --fps N                (default: 25)
  --is-listener true|false   (default: false)
  --hist-id N            (default: 1)
  --user TAG             (default: polito)
  --lon VAL              (default: -98.6)
  --lat VAL              (default: 37.5)

addoutputstream options:
  --acronym ACR              (es. LOWER)                 mandatory
  --name NAME                (es. "low2high")            mandatory
  --pmus "PMU-1,PMU-2,..."   elenco PMU da collegare     mandatory
  --port N                   CommandChannel port (4712 default)
  --fps N                    FramesPerSecond (30 default)
  --nomfreq N                NominalFrequency (60 default)
  --lag MS                   LagTime in ms (3 default)
  --lead MS                  LeadTime in ms (1 default)
  --user TAG                 UpdatedBy/CreatedBy (polito default)

createhistorian options:
  --name NAME                (default: localhistorian)
  --acronym ACR              (default: LOCAL)
  --connstr STR              ConnectionString (default: NULL)
  --mri N                    MeasurementReportingInterval (default: 100000)
  --user TAG                 UpdatedBy/CreatedBy (default: polito)

connectiontopdc options:
  --name NAME                 mandatory
  --acronym ACR               mandatory
  --server                    mandatory
  --pmus                      mandatory

createaccount options:
  --username USER         mandatory
  --password PWD          mandatory
  --firstname NAME       mandatory
  --lastname NAME        mandatory

Examples:
  $0 addpmu --name "Pmu-3" --pod <podname> --ns lower --db lower
  $0 createoutputstream --ns lower --db lower --pod <podname> --acronym LOWER --name low2high  --pmus "PMU-3"
  $0 createhistorian --db higher --ns higher --pod <podname>
  $0 connectiontopdc --ns higher --db higher --name "lowerpdc" --pod <podname> --acronym "LOWER" --server "openpdc-low"  --pmus "PMU-1,PMU-2,PMU-3"
  $0 createaccount --ns higher --db higher --pod <podname> --username polito --password Polito00 --firstname polito --lastname rse
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    addpmu|createoutputstream|createhistorian|connectiontopdc|createaccount|help) SUBCOMMAND="$1"; shift; break;;
    -h|--help) usage_global; exit 0;;
    *) echo "Unknown command: $1"; usage_global; exit 1;;
    esac
done

check_global_params(){
  [[ -z "${CLUSTER_PREFIX:-}" ]] && CLUSTER_PREFIX="cluster1"

  POD="${POD:-${PXC_POD:-}}"
  POD="${POD:-${CLUSTER_PREFIX}-pxc-0}"

  PXC_POD="${PXC_POD:-${POD}}"

  SVC="${SVC:-${HAPROXY_SVC:-${CLUSTER_PREFIX}-haproxy}}"
  HAPROXY_SVC="${HAPROXY_SVC:-${SVC}}"

  SECRET="${SECRET:-${ROOT_SECRET_NAME:-${CLUSTER_PREFIX}-secrets}}"
  ROOT_SECRET_NAME="${ROOT_SECRET_NAME:-${SECRET}}"

  OPENPDC_POD="${OPENPDC_POD:-${POD:-}}"

  if [[ -z "${NS:-}" ]]; then
    echo "Error: namespace not set (use --ns)." >&2
    return 1
  fi

  if ! kubectl get namespace "$NS" >/dev/null 2>&1; then
    echo "Error: Namespace '$NS' does not exist." >&2
    return 1
  fi

  if ! kubectl get pod "$PXC_POD" -n "$NS" >/dev/null 2>&1; then
    echo "Error: Pod '$PXC_POD' does not exist in namespace '$NS'." >&2
    return 1
  fi

  if ! kubectl get svc "$HAPROXY_SVC" -n "$NS" >/dev/null 2>&1; then
    echo "Error: Service '$HAPROXY_SVC' does not exist in namespace '$NS'." >&2
    return 1
  fi

  if [[ -z "${OPENPDC_POD:-}" ]]; then
    echo "Error: openPDC pod not set (use --pod)." >&2
    return 1
  fi
  # set ns to lower for pdc location
  if ! kubectl get pod "$OPENPDC_POD" -n lower >/dev/null 2>&1; then
    echo "Error: Pod '$OPENPDC_POD' does not exist in namespace '$NS'." >&2
    return 1
  fi

  return 0
}

run_mysql() {
  local POD="$1"; shift
  local SVC="$1"; shift
  local ROOTPWD="$1"; shift
  local DB="$1"; shift
  local SQL="$1"

  set +e
  local out rc dup_msg
  out="$(printf "%s" "$SQL" | kubectl exec -i "$POD" -c pxc -n "$NS" -- env MYSQL_PWD="$ROOTPWD" mysql -h "$SVC" -uroot --database "$DB" --batch --silent 2>&1)"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    if printf "%s" "$out" | grep -qiE "Duplicate entry|ERROR 1062"; then
      dup_msg="$(printf "%s" "$out" | grep -iE "Duplicate entry|ERROR 1062" | head -n1 | sed -e 's/^[[:space:]]*//')"
      echo "[❌ERROR] MySQL: duplicate entry detected — ${dup_msg}" >&2
      exit 1
    else
      echo "MySQL ERROR (exit $rc):" >&2
      printf "%s\n" "$out" >&2
      return $rc
    fi
  fi

  return 0
}

addpmu_cmd() {
      
    local FPS=25 PORT=4712 NAME="" ACRONYM="" SERVER=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
      usage_global
      return 0
      ;;
      --ns) NS="$2"; shift 2;;
      --cluster-prefix) CLUSTER_PREFIX="$2"; POD="${CLUSTER_PREFIX}-pxc-0"; SVC="${CLUSTER_PREFIX}-haproxy"; SECRET="${CLUSTER_PREFIX}-secrets"; shift 2;;
      --db) DB_NAME="$2"; shift 2;;
      --pod) OPENPDC_POD="$2"; shift 2;;
      --pxc-pod) POD="$2"; shift 2;;
      --haproxy-svc) SVC="$2"; shift 2;;
      --secret-name) ROOT_SECRET_NAME="$2"; shift 2;;
      --name) NAME="$2"; shift 2;;
      --acronym) ACRONYM="$2"; shift 2;;
      --server) SERVER="$2"; shift 2;;
      --fps) FPS="$2"; shift 2;;
      --port) PORT="$2"; shift 2;;
      *) echo "Argument unknown: $1"; return 1;;
    esac
  done
  
  #if no ns or db or pod, exit
  if [[ -z "$NS" ]]; then
    echo "Error: --ns <namespace> is mandatory."
    return 1
  fi
  if [[ -z "$DB_NAME" ]]; then
    echo "Error: --db <name> is mandatory."
    return 1
  fi
  if [[ -z "$OPENPDC_POD" ]]; then
    echo "Error: --pod <podname> is mandatory."
    return 1
  fi
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

  if [[ -z "$NAME" && -n "$ACRONYM" ]]; then
    # es: PMU-3 -> Pmu-3
    NAME="$(echo "$ACRONYM" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
  fi

  check_global_params || exit 1
  echo "Namespace: $NS"
  echo "DB: $DB_NAME  Pod: $POD  Svc: $SVC"
  echo "Name: $NAME  Acronym: $ACRONYM  Server: $SERVER  FPS: $FPS  Port: $PORT"

  ROOTPWD="$(kubectl get secrets "$SECRET" -n "$NS" -o jsonpath='{.data.root}' | base64 --decode)"
  
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
  '', ${FPS}, 0, 5,
  '', NULL, 0, 1, 10, 5, 5, 1, 1, 0,
  100000, 0, 'polito', NOW(6), 'polito', NOW(6)
);

SET @DeviceID := LAST_INSERT_ID();


INSERT INTO Measurement
  (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex,
   SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled,
   UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES
  (1, @DeviceID, '_${ACRONYM}:AV1', 'E', 7, @NULL, '${ACRONYM}-AV1', 0, 1, 0, 1, '${NAME} Analog Value 1',1, 'polito', NOW(6), 'polito', NOW(6)),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
  (1, @DeviceID, '_${ACRONYM}:F',  NULL, 5,  NULL, '${ACRONYM}-FQ', 0, 1, 0, 1, '${NAME} Frequency',1, 'polito', NOW(6), 'polito', NOW(6)),
  (1, @DeviceID, '_${ACRONYM}:DF', NULL, 6,  NULL, '${ACRONYM}-DF', 0, 1, 0, 1, '${NAME} Frequency Delta (dF/dt)',1, 'polito', NOW(6), 'polito', NOW(6)),
  (1, @DeviceID, '_${ACRONYM}:S',  NULL, 8,  NULL, '${ACRONYM}-SF', 0, 1, 0, 1, '${NAME} Status Flags',1, 'polito', NOW(6), 'polito', NOW(6));

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

INSERT INTO Measurement
  (HistorianID, DeviceID, PointTag, SignalTypeID, PhasorSourceIndex, SignalReference, Description, Enabled)
VALUES
  (1, @DeviceID, 'GPA_${ACRONYM}:QF', 13, NULL, '${ACRONYM}-QF', '${NAME} Time Quality Flags', 1);
INSERT INTO TrackedChange (TableName, PrimaryKeyColumn, PrimaryKeyValue) VALUES ('Device', 'ID', @DeviceID);

INSERT INTO Phasor (DeviceID, Label, Type, Phase, SourceIndex, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn) VALUES (@DeviceID, 'A', 'V', '+', 1, 'polito', timestamp('2025-10-02 13:33:51.738862'), 'polito', timestamp('2025-10-02 13:33:51.738862'));
INSERT INTO Phasor (DeviceID, Label, Type, Phase, SourceIndex, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn) VALUES (@DeviceID, 'B', 'V', '+', 2, 'polito', timestamp('2025-10-02 13:33:51.993151'), 'polito', timestamp('2025-10-02 13:33:51.993151'));
INSERT INTO Phasor (DeviceID, Label, Type, Phase, SourceIndex, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn) VALUES (@DeviceID, 'C', 'V', '+', 3, 'polito', timestamp('2025-10-02 13:33:52.156012'), 'polito', timestamp('2025-10-02 13:33:52.156012'));

INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!PMU:ST1', 11, '${ACRONYM}!PMU-ST1', 'Device statistic for Number of data quality errors reported by device during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                                
INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!PMU:ST2', 11, '${ACRONYM}!PMU-ST2', 'Device statistic for Number of time quality errors reported by device during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                                
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!PMU:ST3', 11, '${ACRONYM}!PMU-ST3', 'Device statistic for Number of device errors reported by device during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                                     
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!PMU:ST4', 11, '${ACRONYM}!PMU-ST4', 'Device statistic for Number of measurements received from device during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                                    
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!PMU:ST5', 11, '${ACRONYM}!PMU-ST5', 'Device statistic for Expected number of measurements received from device during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                           
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!PMU:ST6', 11, '${ACRONYM}!PMU-ST6', 'Device statistic for Number of measurements received while device was reporting errors during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                              
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!PMU:ST7', 11, '${ACRONYM}!PMU-ST7', 'Device statistic for Number of defined measurements (per frame) from device during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                         
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST1', 11, '${ACRONYM}!IS-ST1', 'InputStream statistic for Total number of frames received from input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                           
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST2', 11, '${ACRONYM}!IS-ST2', 'InputStream statistic for Timestamp of last received data frame from input stream.', 1);                                                                                                                                                                                                                                                                                                                                                                                    
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST3', 11, '${ACRONYM}!IS-ST3', 'InputStream statistic for Number of frames that were not received from input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                   
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST4', 11, '${ACRONYM}!IS-ST4', 'InputStream statistic for Number of CRC errors reported from input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                             
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST5', 11, '${ACRONYM}!IS-ST5', 'InputStream statistic for Number of out-of-order frames received from input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                    
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST6', 11, '${ACRONYM}!IS-ST6', 'InputStream statistic for Minimum latency from input stream, in milliseconds, during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                         
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST7', 11, '${ACRONYM}!IS-ST7', 'InputStream statistic for Maximum latency from input stream, in milliseconds, during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                         
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST8', 11, '${ACRONYM}!IS-ST8', 'InputStream statistic for Boolean value representing if input stream was continually connected during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                        
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST9', 11, '${ACRONYM}!IS-ST9', 'InputStream statistic for Boolean value representing if input stream has received (or has cached) a configuration frame during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                               
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST10', 11, '${ACRONYM}!IS-ST10', 'InputStream statistic for Number of configuration changes reported by input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                  
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST11', 11, '${ACRONYM}!IS-ST11', 'InputStream statistic for Number of data frames received from input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                          
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST12', 11, '${ACRONYM}!IS-ST12', 'InputStream statistic for Number of configuration frames received from input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                 
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST13', 11, '${ACRONYM}!IS-ST13', 'InputStream statistic for Number of header frames received from input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                        
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST14', 11, '${ACRONYM}!IS-ST14', 'InputStream statistic for Average latency, in milliseconds, for data received from input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                     
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST15', 11, '${ACRONYM}!IS-ST15', 'InputStream statistic for Frame rate as defined by input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                                     
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST16', 11, '${ACRONYM}!IS-ST16', 'InputStream statistic for Latest actual mean frame rate for data received from input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                         
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST17', 11, '${ACRONYM}!IS-ST17', 'InputStream statistic for Latest actual mean Mbps data rate for data received from input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                     
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST18', 11, '${ACRONYM}!IS-ST18', 'InputStream statistic for Number of data units that were not received at least once from input stream during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                               
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST19', 11, '${ACRONYM}!IS-ST19', 'InputStream statistic for Number of bytes received from the input source during last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                            
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST20', 11, '${ACRONYM}!IS-ST20', 'InputStream statistic for Number of processed measurements reported by the input stream during the lifetime of the input stream.', 1);                                                                                                                                                                                                                                                                                                                                    
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST21', 11, '${ACRONYM}!IS-ST21', 'InputStream statistic for Number of bytes received from the input source during the lifetime of the input stream.', 1);                                                                                                                                                                                                                                                                                                                                                   
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST22', 11, '${ACRONYM}!IS-ST22', 'InputStream statistic for The minimum number of measurements received per second during the last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST23', 11, '${ACRONYM}!IS-ST23', 'InputStream statistic for The maximum number of measurements received per second during the last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST24', 11, '${ACRONYM}!IS-ST24', 'InputStream statistic for The average number of measurements received per second during the last reporting interval.', 1);                                                                                                                                                                                                                                                                                                                                                
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST25', 11, '${ACRONYM}!IS-ST25', 'InputStream statistic for Minimum latency from input stream, in milliseconds, during the lifetime of the input stream.', 1);                                                                                                                                                                                                                                                                                                                                              
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST26', 11, '${ACRONYM}!IS-ST26', 'InputStream statistic for Maximum latency from input stream, in milliseconds, during the lifetime of the input stream.', 1);                                                                                                                                                                                                                                                                                                                                              
 INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, SignalReference, Description, Enabled) VALUES(2, @DeviceID, 'GPA_${ACRONYM}!IS:ST27', 11, '${ACRONYM}!IS-ST27', 'InputStream statistic for Average latency, in milliseconds, for data received from input stream during the lifetime of the input stream.', 1);
EOF
)

#echo "---------- BEGIN SQL ----------"
#printf "%s\n" "$SQL"
#echo "----------- END SQL -----------"

printf "%s" "$SQL" | kubectl exec -i "$POD" -c pxc -n "$NS" -- \
  mysql -h "$SVC" -uroot -p"$ROOTPWD" --database "$DB_NAME" --batch --silent

#run_mysql "$POD" "$SVC" "$ROOTPWD" "$DB_NAME" "$SQL"

sleep 1
echo "🔄 Reloading openPDC configuration..."
  if kubectl exec -i "$OPENPDC_POD" -n "$NS" -c openpdc -- bash -lc \
   "screen -ls | grep -q '\.openpdc' && screen -S openpdc -X stuff $'ReloadConfig\r'" \
   >/dev/null 2>&1; then
  sleep 1
  echo "✅ Configuration successfully reloaded!"
else
  echo "Impossible to send ReloadConfig to '$OPENPDC_POD'." >&2
  echo "Check that openPDC is running inside a screen session called 'openpdc'. Otherwise, run the ReloadConfig via the openPDC Manager." >&2
fi

echo "[OK] PMU '$NAME' ($ACRONYM) successfully added on db '$DB_NAME'."

}


createoutputstream_cmd() {
    FPS=30
    PORT=4712
    NAME=""
    ACRONYM=""
    SERVER=""
    
    local PMUS=""  NOMFREQ=60 LAG=3 LEAD=1 USERTAG="polito" 

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
      usage_global
      return 0
      ;;
      --ns) NS="$2"; shift 2;;
      --cluster-prefix) CLUSTER_PREFIX="$2"; shift 2;;
      --db) DB_NAME="$2"; shift 2;;
      --pod) OPENPDC_POD="$2"; shift 2;;
      --pxc-pod) POD="$2"; shift 2;;
      --haproxy-svc) SVC="$2"; shift 2;;
      --secret-name) ROOT_SECRET_NAME="$2"; shift 2;;
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

  POD="${CLUSTER_PREFIX}-pxc-0"
  SVC="${CLUSTER_PREFIX}-haproxy"
  SECRET="${CLUSTER_PREFIX}-secrets"

  #if not ns, db, pmus, name or acronym, exit
  if [[ -z "$NS" ]]; then
    echo "Error: --ns <namespace> is mandatory."
    return 1
  fi
  if [[ -z "$DB_NAME" ]]; then
    echo "Error: --db <name> is mandatory."
    return 1
  fi
  if [[ -z "$PMUS" ]]; then
    echo "Error: --pmus <list> is mandatory."
    return 1
  fi
  if [[ -z "$NAME" ]]; then
    echo "Error: --name <name> is mandatory."
    return 1
  fi
  if [[ -z "$ACRONYM" ]]; then
    echo "Error: --acronym <acronym> is mandatory."
    return 1
  fi
  if [[ -z "$OPENPDC_POD" ]]; then
    echo "Error: --pod <podname> is mandatory."
    return 1
  fi

  echo "Namespace: $NS"
  echo "DB: $DB_NAME  Pod: $POD  Svc: $SVC"
  echo "OutputStream: $NAME ($ACRONYM)  PMUs: $PMUS  FPS: $FPS  Port: $PORT"

  check_global_params || exit 1
  ROOTPWD="$(kubectl get secrets "$SECRET" -n "$NS" -o jsonpath='{.data.root}' | base64 --decode)"

  SQL=$(cat <<EOF
USE \`${DB_NAME}\`;
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
  IFS=',' read -ra PMU_ARR <<< "$PMUS"
  idcode=1
  for raw in "${PMU_ARR[@]}"; do
    pmu="$(echo "$raw" | xargs)"         # trim
    [[ -z "$pmu" ]] && continue
    pmu_name="$(echo "$pmu" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"

    BLOCK=$(cat <<EOF

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


  
   printf "%s" "$SQL" | kubectl exec -i "$POD" -c pxc -n "$NS" -- \
   mysql -h "$SVC" -uroot -p"$ROOTPWD" --database "$DB_NAME" --batch --silent

   
    #echo "---------- BEGIN SQL ----------"
    #printf "%s\n" "$SQL"
    #echo "----------- END SQL -----------"
  echo "🔄 Reloading openPDC configuration..."
  if kubectl exec -i "$OPENPDC_POD" -n "$NS" -c openpdc -- bash -lc \
   "screen -ls | grep -q '\.openpdc' && screen -S openpdc -X stuff $'ReloadConfig\r'" \
   >/dev/null 2>&1; then
    sleep 1
    echo "✅ Configuration successfully reloaded!"
  else
    echo "Impossible to send ReloadConfig to '$OPENPDC_POD'." >&2
    echo "Check that openPDC is running inside a screen session called 'openpdc'. Otherwise, run the ReloadConfig via the openPDC Manager." >&2
  fi

  echo  "[OK] OutputStream '${NAME}' (${ACRONYM}) successfully created with PMUs: ${PMUS}"
}

createhistorian_cmd() {
  
  local USERTAG="polito"
  local NAME="localhistorian"
  local ACRONYM="LOCAL"
  local CONNSTR=""    
  local MRI=100000    

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage_global; return 0;;
      --ns) NS="$2"; shift 2;;
      --cluster-prefix) CLUSTER_PREFIX="$2"; shift 2;;
      --db) DB_NAME="$2"; shift 2;;
      --pod) OPENPDC_POD="$2"; shift 2;;
      --pxc-pod) POD="$2"; shift 2;;
      --haproxy-svc) SVC="$2"; shift 2;;
      --secret-name) ROOT_SECRET_NAME="$2"; shift 2;;
      --name) NAME="$2"; shift 2;;
      --acronym) ACRONYM="$2"; shift 2;;
      --connstr) CONNSTR="$2"; shift 2;;
      --mri) MRI="$2"; shift 2;;
      *) echo "Argument unknown: $1"; return 1;;
    esac
  done

  #if no ns or db or podname, exit
  if [[ -z "$NS" ]]; then
    echo "Error: --ns <namespace> is mandatory."
    return 1 
  fi
  if [[ -z "$DB_NAME" ]]; then
    echo "Error: --db <name> is mandatory."
    return 1
  fi
  if [[ -z "$OPENPDC_POD" ]]; then
    echo "Error: --pod <podname> is mandatory."
    return 1
  fi
  

  check_global_params || exit 1
  echo "Namespace: $NS"
  echo "DB: $DB_NAME  Pod: $POD  Svc: $SVC"
  echo "Historian: $NAME ($ACRONYM)"
 # --- root pwd dal secret ---
  local ROOTPWD
  ROOTPWD="$(kubectl get secrets "$SECRET" -n "$NS" -o jsonpath='{.data.root}' | base64 --decode)"

  local CONN_SQL="NULL"
  if [[ -n "$CONNSTR" ]]; then
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
  

  printf "%s" "$SQL" | kubectl exec -i "$POD" -c pxc -n "$NS" -- \
    mysql -h "$SVC" -uroot -p"$ROOTPWD" --database "$DB_NAME" --batch --silent

  
#set ns to lower for openPDC pod location
echo "🔄 Reloading openPDC configuration..."
  if kubectl exec -i "$OPENPDC_POD" -n lower -c openpdc -- bash -lc \
   "screen -ls | grep -q '\.openpdc' && screen -S openpdc -X stuff $'ReloadConfig\r'" \
   >/dev/null 2>&1; then
    sleep 1
    echo "✅ Configuration successfully reloaded!"
  else
    echo "Impossible to send ReloadConfig to '$OPENPDC_POD'." >&2
    echo "Check that openPDC is running inside a screen session called 'openpdc'. Otherwise, run the ReloadConfig via the openPDC Manager." >&2
  fi

  echo "[OK] Historian '${NAME}' (${ACRONYM}) successfully created."

}

connectiontopdc_cmd(){
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage_global; return 0;;
      --ns) NS="$2"; shift 2;;
      --cluster-prefix) CLUSTER_PREFIX="$2"; POD="${CLUSTER_PREFIX}-pxc-0"; SVC="${CLUSTER_PREFIX}-haproxy"; SECRET="${CLUSTER_PREFIX}-secrets"; shift 2;;
      --db) DB_NAME="$2"; shift 2;;
      --pod) OPENPDC_POD="$2"; shift 2;;
      --pxc-pod) POD="$2"; shift 2;;
      --haproxy-svc) SVC="$2"; shift 2;;
      --secret-name) ROOT_SECRET_NAME="$2"; shift 2;;
      --name) NAME="$2"; shift 2;;
      --acronym) ACRONYM="$2"; shift 2;;
      --server) SERVER="$2"; shift 2;;
      --pmus) PMUS="$2"; shift 2;;
      *) echo "Argument unknown: $1"; return 1;;
    esac
  done
  
  if [[ -z "$NS" ]]; then
    echo "Error: --ns <namespace> is mandatory."
    return 1 
  fi
  if [[ -z "$DB_NAME" ]]; then
    echo "Error: --db <name> is mandatory."
    return 1
  fi
  if [[ -z "$NAME" ]]; then
    echo "Error: --name <name> is mandatory."
    return 1
  fi
  if [[ -z "$ACRONYM" ]]; then
    echo "Error: --acronym <acronym> is mandatory."
    return 1
  fi
  if [[ -z "$SERVER" ]]; then
    echo "Error: --server <host> is mandatory."
    return 1
  fi
  if [[ -z "$PMUS" ]]; then
    echo "Error: --pmus <list> is mandatory."
    return 1
  fi
  if [[ -z "$OPENPDC_POD" ]]; then
    echo "Error: --pod <podname> is mandatory."
    return 1
  fi
  
  check_global_params || exit 1
  echo "Namespace: $NS"
  echo "DB: $DB_NAME  Pod: $POD  Svc: $SVC"
  echo "Connection name: $NAME ($ACRONYM)"

  local ROOTPWD
  ROOTPWD="$(kubectl get secrets "$SECRET" -n "$NS" -o jsonpath='{.data.root}' | base64 --decode)"

  SQL=$(cat <<EOF
USE \`${DB_NAME}\`;
SET @NULL := NULL;


SET @NodeID := (SELECT ID FROM Node LIMIT 1);
UPDATE Node
SET Settings = CONCAT(
  'RemoteStatusServerConnectionString={server=', '${SERVER}', ':8500; integratedSecurity=true}; ',
  'datapublisherport=6165; ',
  'RealTimeStatisticServiceUrl=http://${SERVER}:6052/historian; ',
  'AlarmServiceUrl=http://${SERVER}:5018/alarmservices'
)
WHERE ID = @NodeID;


SET @ParentUniqueID := UUID();
INSERT INTO Device (
  NodeID, ParentID, UniqueID, Acronym, Name, IsConcentrator, CompanyID, HistorianID,
  AccessID, VendorDeviceID, ProtocolID, Longitude, Latitude, InterconnectionID,
  ConnectionString, TimeZone, FramesPerSecond, TimeAdjustmentTicks, DataLossInterval,
  ContactList, MeasuredLines, LoadOrder, Enabled, AllowedParsingExceptions,
  ParsingExceptionWindow, DelayedConnectionInterval, AllowUseOfCachedConfiguration,
  AutoStartDataParsingSequence, SkipDisableRealTimeData, MeasurementReportingInterval,
  ConnectOndemand, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn
) VALUES (
  @NodeID, @NULL, @ParentUniqueID, '${ACRONYM}', '${NAME}', 1, @NULL, 2,
  0, @NULL, 1, @NULL, @NULL, @NULL,
  CONCAT('port=4712; maxSendQueueSize=-1; server=', '${SERVER}', '; islistener=false; transportprotocol=tcp; interface=0.0.0.0'),
  @NULL, 30, 0, 5,
  @NULL, @NULL, 0, 1, 10,
  5, 5, 1,
  1, 0, 100000,
  0, 'polito', NOW(), 'polito', NOW()
);


SET @ParentID := (SELECT ID FROM Device WHERE UniqueID = @ParentUniqueID);
EOF
)

  IFS=',' read -ra __PMUS_ARR <<< "$PMUS"
  __idx=0
  for __pmu in "${__PMUS_ARR[@]}"; do
    __idx=$((__idx+1))                        
    __label="$(printf '%s' "$__pmu" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    __acronym="$__label"
    __name="$__label"

    __SQL_BLOCK=$(cat <<EOSQL
-- === PMU ${__name} ===
SET @ChildUniqueID := UUID();
INSERT INTO Device (
  NodeID, ParentID, UniqueID, Acronym, Name, IsConcentrator, CompanyID, HistorianID,
  AccessID, VendorDeviceID, ProtocolID, Longitude, Latitude, InterconnectionID,
  ConnectionString, TimeZone, FramesPerSecond, TimeAdjustmentTicks, DataLossInterval,
  ContactList, MeasuredLines, LoadOrder, Enabled, AllowedParsingExceptions,
  ParsingExceptionWindow, DelayedConnectionInterval, AllowUseOfCachedConfiguration,
  AutoStartDataParsingSequence, SkipDisableRealTimeData, MeasurementReportingInterval,
  ConnectOndemand, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn
) VALUES (
  @NodeID, @ParentID, @ChildUniqueID, '${__acronym}', '${__name}', 0, @NULL, 2,
  ${__idx}, @NULL, 1, -98.6, 37.5, @NULL,
  '', @NULL, 30, 0, 5,
  @NULL, @NULL, ${__idx}, 1, 10,
  5, 5, 1,
  1, 0, 100000,
  0, 'polito', NOW(), 'polito', NOW()
);
SET @DeviceID := (SELECT ID FROM Device WHERE UniqueID = @ChildUniqueID);

INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (2, @DeviceID, '_${__acronym}:AV1', @NULL, 7, @NULL, '${__acronym}-AV1', 0, 1, 0, 1, '${__name} Analog Value 1', 1, 'polito', NOW(), 'polito', NOW());

INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (2, @DeviceID, '_${__acronym}:F', @NULL, 5, @NULL, '${__acronym}-FQ', 0, 1, 0, 1, '${__name} Frequency', 1, 'polito', NOW(), 'polito', NOW());

INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (2, @DeviceID, '_${__acronym}:DF', @NULL, 6, @NULL, '${__acronym}-DF', 0, 1, 0, 1, '${__name} Frequency Delta (dF/dt)', 1, 'polito', NOW(), 'polito', NOW());

INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (2, @DeviceID, '_${__acronym}:S', @NULL, 8, @NULL, '${__acronym}-SF', 0, 1, 0, 1, '${__name} Status Flags', 1, 'polito', NOW(), 'polito', NOW());

-- Phasor A/B/C con misure Magnitude (3) e Phase Angle (4)
INSERT INTO Phasor (DeviceID, Label, Type, Phase, SourceIndex, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (@DeviceID, 'A +SV', 'V', '+', 1, 'polito', NOW(), 'polito', NOW());
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (2, @DeviceID, '_${__acronym}-PM1:V', @NULL, 3, 1, '${__acronym}-PM1', 0, 1, 0, 1, '${__name} A +SV  + Voltage Magnitude', 1, 'polito', NOW(), 'polito', NOW());
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (2, @DeviceID, '_${__acronym}-PA1:VH', @NULL, 4, 1, '${__acronym}-PA1', 0, 1, 0, 1, '${__name} A +SV  + Voltage Phase Angle', 1, 'polito', NOW(), 'polito', NOW());

INSERT INTO Phasor (DeviceID, Label, Type, Phase, SourceIndex, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (@DeviceID, 'B +SV', 'V', '+', 2, 'polito', NOW(), 'polito', NOW());
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (2, @DeviceID, '_${__acronym}-PM2:V', @NULL, 3, 2, '${__acronym}-PM2', 0, 1, 0, 1, '${__name} B +SV  + Voltage Magnitude', 1, 'polito', NOW(), 'polito', NOW());
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (2, @DeviceID, '_${__acronym}-PA2:VH', @NULL, 4, 2, '${__acronym}-PA2', 0, 1, 0, 1, '${__name} B +SV  + Voltage Phase Angle', 1, 'polito', NOW(), 'polito', NOW());

INSERT INTO Phasor (DeviceID, Label, Type, Phase, SourceIndex, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (@DeviceID, 'C +SV', 'V', '+', 3, 'polito', NOW(), 'polito', NOW());
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (2, @DeviceID, '_${__acronym}-PM3:V', @NULL, 3, 3, '${__acronym}-PM3', 0, 1, 0, 1, '${__name} C +SV  + Voltage Magnitude', 1, 'polito', NOW(), 'polito', NOW());
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (2, @DeviceID, '_${__acronym}-PA3:VH', @NULL, 4, 3, '${__acronym}-PA3', 0, 1, 0, 1, '${__name} C +SV  + Voltage Phase Angle', 1, 'polito', NOW(), 'polito', NOW());
EOSQL
    )

    SQL="${SQL}\n${__SQL_BLOCK}\n"
  done

  
   #echo "---------- BEGIN SQL ----------"
   # printf "%s\n" "$SQL"
  #echo "----------- END SQL -----------"

  
  printf "%s" "$SQL" | kubectl exec -i "$POD" -c pxc -n "$NS" -- \
    mysql -h "$SVC" -uroot -p"$ROOTPWD" --database "$DB_NAME" --batch --silent

#set ns to lower for openPDC pod location
  echo "🔄 Reloading openPDC configuration..."
  if kubectl exec -i "$OPENPDC_POD" -n lower -c openpdc -- bash -lc \
   "screen -ls | grep -q '\.openpdc' && screen -S openpdc -X stuff $'ReloadConfig\r'" \
   >/dev/null 2>&1; then
    sleep 1
    echo "✅ Configuration successfully reloaded!"
  else
    echo "Impossible to send ReloadConfig to '$OPENPDC_POD'." >&2
    echo "Check that openPDC is running inside a screen session called 'openpdc'. Otherwise, run the ReloadConfig via the openPDC Manager." >&2
  fi

  echo "[OK] Connection '${NAME}' to PDC server '${SERVER}' successfully created with PMUs: ${PMUS}"

}
createaccount_cmd() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage_global; return 0;;
      --ns) NS="$2"; shift 2;;
      --cluster-prefix) CLUSTER_PREFIX="$2"; POD="${CLUSTER_PREFIX}-pxc-0"; SVC="${CLUSTER_PREFIX}-haproxy"; SECRET="${CLUSTER_PREFIX}-secrets"; shift 2;;
      --db) DB_NAME="$2"; shift 2;;
      --pod) OPENPDC_POD="$2"; shift 2;;
      --pxc-pod) POD="$2"; shift 2;;
      --haproxy-svc) SVC="$2"; shift 2;;
      --username) USERNAME="$2"; shift 2;;
      --password) PASSWORD="$2"; shift 2;;
      --firstname) FIRSTNAME="$2"; shift 2;;
      --lastname) LASTNAME="$2"; shift 2;;
      *) echo "Argument unknown: $1"; return 1;;
    esac
  done

  if [[ -z "$NS" ]]; then
    echo "Error: --ns <namespace> is mandatory."
    return 1 
  fi
  if [[ -z "$DB_NAME" ]]; then
    echo "Error: --db <name> is mandatory."
    return 1
  fi
  if [[ -z "$USERNAME" ]]; then
    echo "Error: --username <name> is mandatory."
    return 1
  fi
  if [[ -z "$PASSWORD" ]]; then
    echo "Error: --password <name> is mandatory."
    return 1
  fi
  if [[ -z "$FIRSTNAME" ]]; then
    echo "Error: --firstname <name> is mandatory."
    return 1
  fi
  if [[ -z "$LASTNAME" ]]; then
    echo "Error: --lastname <name> is mandatory."
    return 1
  fi
  if [[ -z "$OPENPDC_POD" ]]; then
    echo "Error: --pod <podname> is mandatory."
    return 1
  fi

  check_global_params || exit 1
  echo "Namespace: $NS"
  echo "DB: $DB_NAME  Pod: $POD  Svc: $SVC"
  echo "Creating user account: $USERNAME ($FIRSTNAME $LASTNAME)"
  local ROOTPWD
  ROOTPWD="$(kubectl get secrets "$SECRET" -n "$NS" -o jsonpath='{.data.root}' | base64 --decode)"

  esc() { printf "%s" "$1" | sed "s/'/''/g"; }
  UESC=$(esc "$USERNAME")
  FESC=$(esc "$FIRSTNAME")
  LESC=$(esc "$LASTNAME")
  PHESC='SpGxVC2T8Hur/HfkWGEklBRNvlX+07gnGuLg7qC6qy0='

  # ---------- Blocco SQL (variabili espanse) ----------
  SQL=$(cat <<SQL_EOF
SET NAMES utf8mb4;
SET character_set_results = NULL;

-- Prende un Node valido (il primo disponibile)
SET @NodeID := (SELECT ID FROM Node LIMIT 1);

-- 1) RUOLI (idempotente)
INSERT INTO ApplicationRole (ID, Name, Description, NodeID, UpdatedBy, CreatedBy)
SELECT UUID(), 'Administrator', 'Administrator Role', @NodeID, 'CLI', 'CLI'
WHERE NOT EXISTS (SELECT 1 FROM ApplicationRole WHERE Name='Administrator' AND NodeID=@NodeID);

INSERT INTO ApplicationRole (ID, Name, Description, NodeID, UpdatedBy, CreatedBy)
SELECT UUID(), 'Editor', 'Editor Role', @NodeID, 'CLI', 'CLI'
WHERE NOT EXISTS (SELECT 1 FROM ApplicationRole WHERE Name='Editor' AND NodeID=@NodeID);

INSERT INTO ApplicationRole (ID, Name, Description, NodeID, UpdatedBy, CreatedBy)
SELECT UUID(), 'Viewer', 'Viewer Role', @NodeID, 'CLI', 'CLI'
WHERE NOT EXISTS (SELECT 1 FROM ApplicationRole WHERE Name='Viewer' AND NodeID=@NodeID);

-- 2) UTENTE con AUTENTICAZIONE DB (UseADAuthentication = 0)
INSERT INTO UserAccount (ID, Name, Password, FirstName, LastName, DefaultNodeID, UseADAuthentication, CreatedBy, UpdatedBy)
SELECT UUID(), '${UESC}', '${PHESC}', '${FESC}', '${LESC}', @NodeID, 0, 'CLI', 'CLI'
WHERE NOT EXISTS (SELECT 1 FROM UserAccount WHERE Name='${UESC}');

-- 3) Collega l'utente al ruolo Administrator (idempotente)
INSERT INTO ApplicationRoleUserAccount (ApplicationRoleID, UserAccountID)
SELECT ar.ID, ua.ID
FROM ApplicationRole ar
JOIN UserAccount ua ON ua.Name='${UESC}'
WHERE ar.Name='Administrator' AND ar.NodeID=@NodeID
  AND NOT EXISTS (
    SELECT 1
    FROM ApplicationRoleUserAccount x
    WHERE x.ApplicationRoleID = ar.ID AND x.UserAccountID = ua.ID
  );

-- 4) (Facoltativo) registra un accesso positivo a log
INSERT INTO AccessLog (UserName, AccessGranted) VALUES ('${UESC}', 1);
SQL_EOF
)

#echo "---------- BEGIN SQL ----------"
#    printf "%s\n" "$SQL"
#  echo "----------- END SQL -----------"

  
  printf "%s" "$SQL" | kubectl exec -i "$POD" -c pxc -n "$NS" -- \
    mysql -h "$SVC" -uroot -p"$ROOTPWD" --database "$DB_NAME" --batch --silent

  #set db to lower for pdc location
  echo "🔄 Reloading openPDC configuration..."
  if kubectl exec -i "$OPENPDC_POD" -n lower -c openpdc -- bash -lc \
   "screen -ls | grep -q '\.openpdc' && screen -S openpdc -X stuff $'ReloadConfig\r'" \
   >/dev/null 2>&1; then
   sleep 1
    echo "✅ Configuration successfully reloaded!"
  else
    echo "Impossible to send ReloadConfig to '$OPENPDC_POD'." >&2
    echo "Check that openPDC is running inside a screen session called 'openpdc'. Otherwise, run the ReloadConfig via the openPDC Manager." >&2
  fi

  echo "[OK] User account '${USERNAME}' successfully created."

}
case "$SUBCOMMAND" in
  help) usage_global;;
  addpmu) addpmu_cmd "$@";;
  createoutputstream) createoutputstream_cmd "${GLOBAL_ARGS[@]}" "$@";;
  createhistorian) createhistorian_cmd "${GLOBAL_ARGS[@]}" "$@";;
  connectiontopdc) connectiontopdc_cmd "${GLOBAL_ARGS[@]}" "$@";;
  createaccount) createaccount_cmd "${GLOBAL_ARGS[@]}" "$@";;
  *) usage_global; exit 1;;
esac