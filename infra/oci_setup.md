# OCI Deployment Guide

This guide documents how to deploy the Systematic Equity Engine on Oracle Cloud Infrastructure (OCI) Always Free tier.

## Prerequisites

- OCI VM instance (Ubuntu 22.04, Ampere A1)
- Public IP address of your VM
- SSH access configured
- Private key at: `/Users/gaurangj/Documents/CascadeProjects/windsurf-project/ssh-key-2026-01-22.key`

## Variables (Replace with your values)

```bash
export OCI_IP="YOUR_VM_PUBLIC_IP"
export OCI_USER="ubuntu"  # or your SSH username
export SSH_KEY="/Users/gaurangj/Documents/CascadeProjects/windsurf-project/ssh-key-2026-01-22.key"
```

## 1. Initial Server Setup

### Connect to VM

```bash
ssh -i $SSH_KEY $OCI_USER@$OCI_IP
```

### Install Python and dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+ and pip
sudo apt install -y python3 python3-pip python3-venv git

# Verify installation
python3 --version  # Should be 3.10+
```

## 2. Clone Repository

```bash
# Clone the repository
cd ~
git clone https://github.com/gaurangjain95/systematic-equity-engine.git
cd systematic-equity-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Configure Firewall

OCI uses security lists in addition to iptables. You need to configure both.

### OCI Console - Security List Rules

1. Go to OCI Console → Networking → Virtual Cloud Networks
2. Select your VCN → Security Lists → Default Security List
3. Add Ingress Rules:
   - **API (FastAPI)**: Source CIDR `0.0.0.0/0`, Protocol TCP, Destination Port `8000`
   - **Dashboard (Streamlit)**: Source CIDR `0.0.0.0/0`, Protocol TCP, Destination Port `8501`

### Ubuntu iptables

```bash
# Allow API port
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT

# Allow Streamlit port
sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT

# Save rules
sudo apt install -y iptables-persistent
sudo netfilter-persistent save
```

## 4. Running Services

### Option A: Using tmux (Recommended for development)

```bash
# Install tmux
sudo apt install -y tmux

# Start a new session
tmux new -s equity-engine

# Start API server (in first pane)
cd ~/systematic-equity-engine
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Split pane (Ctrl+b, then %)
# Start Streamlit (in second pane)
cd ~/systematic-equity-engine
source venv/bin/activate
streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0

# Detach from session: Ctrl+b, then d
# Reattach later: tmux attach -t equity-engine
```

### Option B: Using systemd (Recommended for production)

#### Create API service

```bash
sudo tee /etc/systemd/system/equity-api.service << 'EOF'
[Unit]
Description=Systematic Equity Engine API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/systematic-equity-engine
Environment="PATH=/home/ubuntu/systematic-equity-engine/venv/bin"
ExecStart=/home/ubuntu/systematic-equity-engine/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

#### Create Streamlit service

```bash
sudo tee /etc/systemd/system/equity-dashboard.service << 'EOF'
[Unit]
Description=Systematic Equity Engine Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/systematic-equity-engine
Environment="PATH=/home/ubuntu/systematic-equity-engine/venv/bin"
ExecStart=/home/ubuntu/systematic-equity-engine/venv/bin/streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

#### Enable and start services

```bash
sudo systemctl daemon-reload
sudo systemctl enable equity-api equity-dashboard
sudo systemctl start equity-api equity-dashboard

# Check status
sudo systemctl status equity-api
sudo systemctl status equity-dashboard

# View logs
sudo journalctl -u equity-api -f
sudo journalctl -u equity-dashboard -f
```

## 5. Verify Deployment

### Test API

```bash
# From your local machine
curl http://$OCI_IP:8000/health
curl http://$OCI_IP:8000/portfolio
curl http://$OCI_IP:8000/performance
```

### Access Dashboard

Open in browser: `http://YOUR_VM_PUBLIC_IP:8501`

## 6. Updating the Application

```bash
# SSH into VM
ssh -i $SSH_KEY $OCI_USER@$OCI_IP

# Pull latest changes
cd ~/systematic-equity-engine
git pull origin main

# Restart services (if using systemd)
sudo systemctl restart equity-api equity-dashboard

# Or if using tmux, manually restart in each pane
```

## 7. Running Strategy Scripts

```bash
# SSH into VM
ssh -i $SSH_KEY $OCI_USER@$OCI_IP

# Activate environment
cd ~/systematic-equity-engine
source venv/bin/activate

# Run weekly rebalance
python -m scripts.run_weekly --as-of-date 2024-01-15

# Run monthly rebalance
python -m scripts.run_monthly --as-of-date 2024-01-15
```

## Troubleshooting

### Services not starting

```bash
# Check detailed logs
sudo journalctl -u equity-api -n 50 --no-pager
sudo journalctl -u equity-dashboard -n 50 --no-pager
```

### Port not accessible

```bash
# Check if service is listening
sudo netstat -tlnp | grep -E '8000|8501'

# Check iptables
sudo iptables -L -n | grep -E '8000|8501'

# Check OCI Security List (must be done in OCI Console)
```

### Python import errors

```bash
# Ensure you're in the right directory and venv is activated
cd ~/systematic-equity-engine
source venv/bin/activate
pip install -r requirements.txt
```

## Security Notes

- This setup has no authentication. Add authentication before exposing to public internet.
- Consider using nginx as a reverse proxy for production.
- Keep your private SSH key secure and never commit it to git.
- Regularly update system packages: `sudo apt update && sudo apt upgrade`
