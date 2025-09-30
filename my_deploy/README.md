# Deploy HA MySQL with Percona and OpenPDC

This guide explains how to deploy a **High Availability MySQL cluster** using **Percona XtraDB Cluster Operator**, and then deploy **OpenPDC** on top of it.

---

## 1. Clone the Percona repository
```
git clone https://github.com/percona/percona-xtradb-cluster-operator.git
cd percona-xtradb-cluster-operator
```

## 2. Apply Custom Resource Definitions (CRDs)
CRDs define the custom resources used by the operator.
```
kubectl apply -f deploy/crd.yaml
```

## 3. Create a dedicated namespace
Choose a namespace (e.g., higher or lower):
```
kubectl create ns <namespace>
```

## 4. Apply RBAC configuration
Grant the required permissions to the operator in the chosen namespace:
```
kubectl apply -n <namespace> -f deploy/rbac.yaml
```

## 5. Deploy the operator
The operator manages the lifecycle of the Percona XtraDB Cluster.
```
kubectl apply -n <namespace> -f deploy/operator.yaml
```

## 6. Create Secrets
Secrets define the database credentials.
```
kubectl apply -n <namespace> -f deploy/secrets.yaml
```

## 7. Deploy the Percona Cluster (CR)
The cr.yaml resource describes the Percona XtraDB Cluster instance.
```
kubectl apply -n <namespace> -f deploy/cr.yaml
```

## 8. Deploy the OpenPDC
Once the Percona cluster is ready, deploy OpenPDC in the same namespace:
```
kubectl apply -n <namespace> -f openpdc.yaml
```

## 🔎 Notes:
1) Replace <namespace> with the namespace you created (e.g., higher or lower).
2) Make sure the nodePort values in your OpenPDC services are unique if deploying multiple instances.






