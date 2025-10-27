# Create and configure k3d

Create a dedicated Docker network for k3d clusters:
```bash
docker network create mc-net
```
Create a k3d cluster without Traefik on the created network:
```bash
k3d cluster create cluster-1 --servers 1 --agents 0 --k3s-arg "--disable=traefik@server:*" --network mc-net
```
Retrieve the IP address of the server node:
```bash
docker inspect -f '{{ (index .NetworkSettings.Networks "mc-net").IPAddress }}' k3d-cluster-5-server-0
```
Run a temporary container to inspect network settings:
```bash
docker run --rm --net container:k3d-cluster-5-server-0 nicolaka/netshoot ip -o -4 addr show
```

Set network latency on the server node (replace `500ms` with desired latency):
```bash
docker run --rm --privileged --net container:k3d-cluster-5-server-0 \
  nicolaka/netshoot tc qdisc replace dev eth0 root netem delay 500ms
```
Try pinging another container in the same network to verify latency 
```bash
docker run --rm --net container:k3d-cluster-5-server-0 nicolaka/netshoot ping -c 5 172.18.0.14
```
Remove the latency configuration when done:
```bash
docker run --rm --privileged --net container:k3d-cluster-5-server-0 \
  nicolaka/netshoot tc qdisc del dev eth0 root
```