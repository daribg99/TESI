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
DB_NAME="openPDC"             # --db
PXC_POD=""                    # --pxc-pod (default: <cluster_prefix>-pxc-0)
HAPROXY_SVC=""                # --haproxy-svc (default: <cluster_prefix>-haproxy)
ROOT_SECRET_NAME=""           # --secret-name (default: <cluster_prefix>-secrets)
MYSQL_HOST=""                 # opzionale override host MySQL (salta k8s)
MYSQL_USER="root"             # --mysql-user
MYSQL_PASS=""                 # --mysql-pass (se vuoto e in k8s, prende dal secret)
DRY_RUN="false"               # --dry-run

# ---------- Helpers ----------
die() { echo "ERRORE: $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }
ok()   { echo "[OK] $*"; }

usage_global() {
  cat <<USAGE
Uso:
  $0 <subcommand> [opzioni globali] [opzioni subcommand]

Subcommands:
  addpmu            Crea un Device (PMU) e le sue Measurement
  addoutputstream   (placeholder) Crea un output stream
  help              Mostra questo help

Opzioni globali:
  --ns NAMESPACE                     (OBBLIGATORIO)
  --cluster-prefix NAME              (default: cluster1)
  --db NAME                          (default: openPDC)
  --pxc-pod NAME                     (default: <cluster>-pxc-0)
  --haproxy-svc NAME                 (default: <cluster>-haproxy)
  --secret-name NAME                 (default: <cluster>-secrets)
  --mysql-host HOST                  (se impostato, usa connessione diretta e salta k8s)
  --mysql-user USER                  (default: root)
  --mysql-pass PASS                  (se vuoto: da secret in k8s)
  --dry-run                          Mostra solo l'SQL

Esempi:
  $0 addpmu --ns demo --acronym PMU-3 --name "Pmu-3" --server pmu-3
  $0 help
USAGE
}

# parse only global opts until troviamo il subcommand
GLOBAL_ARGS=()
SUBCOMMAND=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    addpmu|addoutputstream|help) SUBCOMMAND="$1"; shift; break;;
    --ns) NS="$2"; shift 2;;
    --cluster-prefix) CLUSTER_PREFIX="$2"; shift 2;;
    --db) DB_NAME="$2"; shift 2;;
    --pxc-pod) PXC_POD="$2"; shift 2;;
    --haproxy-svc) HAPROXY_SVC="$2"; shift 2;;
    --secret-name) ROOT_SECRET_NAME="$2"; shift 2;;
    --mysql-host) MYSQL_HOST="$2"; shift 2;;
    --mysql-user) MYSQL_USER="$2"; shift 2;;
    --mysql-pass) MYSQL_PASS="$2"; shift 2;;
    --dry-run) DRY_RUN="true"; shift;;
    -h|--help) usage_global; exit 0;;
    *) GLOBAL_ARGS+=("$1"); shift;;
  esac
done

[[ -z "${SUBCOMMAND}" ]] && { usage_global; exit 1; }

# defaults derivati
[[ -z "$PXC_POD" ]] && PXC_POD="${CLUSTER_PREFIX}-pxc-0"
[[ -z "$HAPROXY_SVC" ]] && HAPROXY_SVC="${CLUSTER_PREFIX}-haproxy"
[[ -z "$ROOT_SECRET_NAME" ]] && ROOT_SECRET_NAME="${CLUSTER_PREFIX}-secrets"

# ---------- SQL escaping (solo apici singoli) ----------
sql_escape() { local s="${1//\'/\'\'}"; echo "$s"; }

# ---------- MySQL executor (k8s o diretto) ----------
mysql_exec() {
  local sql="$1"
  if [[ -n "$MYSQL_HOST" ]]; then
    # connessione diretta
    [[ -z "$MYSQL_PASS" ]] && die "Connessione diretta: serve --mysql-pass"
    mysql -h "$MYSQL_HOST" -u"$MYSQL_USER" -p"$MYSQL_PASS" --protocol=tcp --database "$DB_NAME" --batch --silent -e "$sql"
  else
    # via k8s
    command -v kubectl >/dev/null || die "kubectl non trovato"
    if [[ -z "$MYSQL_PASS" ]]; then
      [[ -z "$NS" ]] && die "Serve --ns per recuperare la password dal secret"
      MYSQL_PASS="$(kubectl get secret "$ROOT_SECRET_NAME" -n "$NS" -o jsonpath='{.data.root}' | base64 --decode || true)"
      [[ -z "$MYSQL_PASS" ]] && die "Password non trovata nel secret ${ROOT_SECRET_NAME} nel ns ${NS}"
    fi
    kubectl exec -i "$PXC_POD" -c pxc -n "$NS" -- \
      mysql -h "$HAPROXY_SVC" -u"$MYSQL_USER" "-p${MYSQL_PASS}" --protocol=tcp --database "$DB_NAME" --batch --silent -e "$sql"
  fi
}

mysql_heredoc() {
  local sql="$1"
  if [[ -n "$MYSQL_HOST" ]]; then
    [[ -z "$MYSQL_PASS" ]] && die "Connessione diretta: serve --mysql-pass"
    mysql -h "$MYSQL_HOST" -u"$MYSQL_USER" -p"$MYSQL_PASS" --protocol=tcp --database "$DB_NAME" --batch --silent <<EOF
$sql
EOF
  else
    command -v kubectl >/dev/null || die "kubectl non trovato"
    if [[ -z "$MYSQL_PASS" ]]; then
      [[ -z "$NS" ]] && die "Serve --ns per recuperare la password dal secret"
      MYSQL_PASS="$(kubectl get secret "$ROOT_SECRET_NAME" -n "$NS" -o jsonpath='{.data.root}' | base64 --decode || true)"
      [[ -z "$MYSQL_PASS" ]] && die "Password non trovata nel secret ${ROOT_SECRET_NAME} nel ns ${NS}"
    fi
    kubectl exec -i "$PXC_POD" -c pxc -n "$NS" -- \
      mysql -h "$HAPROXY_SVC" -u"$MYSQL_USER" "-p${MYSQL_PASS}" --protocol=tcp --database "$DB_NAME" --batch --silent <<EOF
$sql
EOF
  fi
}

# ======================================================
# SUBCOMMAND: addpmu
# ======================================================
addpmu_usage() {
  cat <<USAGE
Uso:
  $0 addpmu --ns <namespace> --acronym <PMU-ACR> --name <Nome> --server <host/ip> [opzioni] [opzioni globali]

Opzioni addpmu:
  --acronym ACR          (es. PMU-3)  OBBLIGATORIO
  --name NAME            (es. "Pmu-3") OBBLIGATORIO
  --server HOST          (endpoint PMU) OBBLIGATORIO
  --port N               (default: 4712)
  --fps N                (default: 25)
  --is-listener true|false   (default: false)
  --hist-id N            (default: 1)
  --user TAG             (default: polito)
  --lon VAL              (default: -98.6)
  --lat VAL              (default: 37.5)
  --force                Non fallire se esiste già un device con stesso Acronym (aggiorno? no: skip)

Esempio:
  $0 addpmu --ns demo --acronym PMU-3 --name "Pmu-3" --server pmu-3 --port 4712 --fps 25
USAGE
}

addpmu_cmd() {
  local ACRONYM="" NAME="" SERVER="" PORT="4712" FPS="25" IS_LISTENER="false" HIST_ID="1"
  local USER_TAG="polito" LON="-98.6" LAT="37.5" FORCE="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --acronym) ACRONYM="$2"; shift 2;;
      --name) NAME="$2"; shift 2;;
      --server) SERVER="$2"; shift 2;;
      --port) PORT="$2"; shift 2;;
      --fps) FPS="$2"; shift 2;;
      --is-listener) IS_LISTENER="$2"; shift 2;;
      --hist-id) HIST_ID="$2"; shift 2;;
      --user) USER_TAG="$2"; shift 2;;
      --lon) LON="$2"; shift 2;;
      --lat) LAT="$2"; shift 2;;
      --force) FORCE="true"; shift;;
      -h|--help) addpmu_usage; exit 0;;
      *) echo "Argomento sconosciuto per addpmu: $1"; addpmu_usage; exit 1;;
    esac
  done

  [[ -z "$NS" && -z "$MYSQL_HOST" ]] && die "Specifica --ns (o usa --mysql-host per connessione diretta)"
  [[ -z "$ACRONYM" || -z "$NAME" || -z "$SERVER" ]] && { addpmu_usage; exit 1; }

  local ACRONYM_SQL NAME_SQL SERVER_SQL USER_SQL
  ACRONYM_SQL="$(sql_escape "$ACRONYM")"
  NAME_SQL="$(sql_escape "$NAME")"
  SERVER_SQL="$(sql_escape "$SERVER")"
  USER_SQL="$(sql_escape "$USER_TAG")"

  local CONN_STR="port=${PORT}; maxSendQueueSize=-1; server=${SERVER_SQL}; islistener=${IS_LISTENER}; transportprotocol=tcp; interface=0.0.0.0"
  local CONN_SQL="$(sql_escape "$CONN_STR")"

  # Esistenza device con stesso Acronym?
  local EXIST_ID=""
  EXIST_ID="$(mysql_exec "SELECT ID FROM Device WHERE Acronym='${ACRONYM_SQL}' LIMIT 1;" || true)"
  if [[ -n "$EXIST_ID" ]]; then
    if [[ "$FORCE" == "true" ]]; then
      info "Device con Acronym ${ACRONYM} già esiste (ID=${EXIST_ID}). Procedo comunque (creerai un duplicato)."
    else
      die "Esiste già un Device con Acronym='${ACRONYM}' (ID=${EXIST_ID}). Usa --force per ignorare."
    fi
  fi

  # Costruisco SQL (come da tue query; rimosso duplicato su :F e @NULL -> NULL)
  read -r -d '' SQL_PAYLOAD <<SQL
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
  '${ACRONYM_SQL}', '${NAME_SQL}', 0, NULL, ${HIST_ID},
  @AccessID, NULL, 1, ${LON}, ${LAT}, NULL,
  '${CONN_SQL}',
  NULL, ${FPS}, 0, 5,
  NULL, NULL, 0, 1, 10, 5, 5, 1, 1, 0,
  100000, 0, '${USER_SQL}', NOW(6), '${USER_SQL}', NOW(6)
);

SET @DeviceID := LAST_INSERT_ID();

-- Frequency
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (${HIST_ID}, @DeviceID, '_${ACRONYM_SQL}:F', NULL, 5, NULL, '${ACRONYM_SQL}-FQ', 0, 1, 0, 1, '${NAME_SQL} Frequency', 1, '${USER_SQL}', NOW(6), '${USER_SQL}', NOW(6));

-- dF/dt
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (${HIST_ID}, @DeviceID, '_${ACRONYM_SQL}:DF', NULL, 6, NULL, '${ACRONYM_SQL}-DF', 0, 1, 0, 1, '${NAME_SQL} Frequency Delta (dF/dt)', 1, '${USER_SQL}', NOW(6), '${USER_SQL}', NOW(6));

-- Status Flags
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (${HIST_ID}, @DeviceID, '_${ACRONYM_SQL}:S', NULL, 8, NULL, '${ACRONYM_SQL}-SF', 0, 1, 0, 1, '${NAME_SQL} Status Flags', 1, '${USER_SQL}', NOW(6), '${USER_SQL}', NOW(6));

-- Phasor A
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (${HIST_ID}, @DeviceID, '_${ACRONYM_SQL}-PM1:V',  NULL, 3, 1, '${ACRONYM_SQL}-PM1', 0, 1, 0, 1, '${NAME_SQL} A  + Voltage Magnitude', 1, '${USER_SQL}', NOW(6), '${USER_SQL}', NOW(6));
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (${HIST_ID}, @DeviceID, '_${ACRONYM_SQL}-PA1:VH', NULL, 4, 1, '${ACRONYM_SQL}-PA1', 0, 1, 0, 1, '${NAME_SQL} A  + Voltage Phase Angle', 1, '${USER_SQL}', NOW(6), '${USER_SQL}', NOW(6));

-- Phasor B
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (${HIST_ID}, @DeviceID, '_${ACRONYM_SQL}-PM2:V',  NULL, 3, 2, '${ACRONYM_SQL}-PM2', 0, 1, 0, 1, '${NAME_SQL} B  + Voltage Magnitude', 1, '${USER_SQL}', NOW(6), '${USER_SQL}', NOW(6));
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (${HIST_ID}, @DeviceID, '_${ACRONYM_SQL}-PA2:VH', NULL, 4, 2, '${ACRONYM_SQL}-PA2', 0, 1, 0, 1, '${NAME_SQL} B  + Voltage Phase Angle', 1, '${USER_SQL}', NOW(6), '${USER_SQL}', NOW(6));

-- Phasor C
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (${HIST_ID}, @DeviceID, '_${ACRONYM_SQL}-PM3:V',  NULL, 3, 3, '${ACRONYM_SQL}-PM3', 0, 1, 0, 1, '${NAME_SQL} C  + Voltage Magnitude', 1, '${USER_SQL}', NOW(6), '${USER_SQL}', NOW(6));
INSERT INTO Measurement (HistorianID, DeviceID, PointTag, AlternateTag, SignalTypeID, PhasorSourceIndex, SignalReference, Adder, Multiplier, Subscribed, Internal, Description, Enabled, UpdatedBy, UpdatedOn, CreatedBy, CreatedOn)
VALUES (${HIST_ID}, @DeviceID, '_${ACRONYM_SQL}-PA3:VH', NULL, 4, 3, '${ACRONYM_SQL}-PA3', 0, 1, 0, 1, '${NAME_SQL} C  + Voltage Phase Angle', 1, '${USER_SQL}', NOW(6), '${USER_SQL}', NOW(6));

-- Quality Flags (QF)
INSERT INTO Measurement(HistorianID, DeviceID, PointTag, SignalTypeID, PhasorSourceIndex, SignalReference, Description, Enabled)
VALUES(${HIST_ID}, @DeviceID, 'GPA_${ACRONYM_SQL}:QF', 13, NULL, '${ACRONYM_SQL}-QF', '${NAME_SQL} Time Quality Flags', 1);
SQL

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "----- SQL che verrebbe eseguito -----"
    echo "$SQL_PAYLOAD"
    exit 0
  fi

  info "Eseguo inserimento PMU ${ACRONYM} su DB ${DB_NAME}..."
  mysql_heredoc "$SQL_PAYLOAD"
  ok "Device ${ACRONYM} creato e misure inserite."
}

# ======================================================
# SUBCOMMAND: addoutputstream (placeholder)
# ======================================================
addoutputstream_cmd() {
  cat <<'MSG'
[TODO] addoutputstream: segnaposto.
L'idea:
  - creare row in OutputStream (o tabella equivalente) + mapping delle misure da sorgente a destinazione
  - parametri attesi: --name --acronym --target-connstring --format --fps --enabled ecc.
Se vuoi, ti preparo lo schema/SQL esatto in base alla tabella reale del tuo DB.
MSG
}

# ======================================================
# DISPATCH
# ======================================================
case "$SUBCOMMAND" in
  help) usage_global;;
  addpmu) addpmu_cmd "$@";;
  addoutputstream) addoutputstream_cmd "$@";;
  *) usage_global; exit 1;;
esac
