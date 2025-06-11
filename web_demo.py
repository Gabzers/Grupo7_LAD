import pandas as pd
import numpy as np
from flask import Flask, render_template_string, jsonify
from sklearn.preprocessing import StandardScaler
from RandomForestAnalysis import RandomForestAnalysis
import random
import threading
import time
import plotly
import plotly.graph_objs as go
import json

# Carregar variáveis do dataset
df = pd.read_csv("RT_IOT2022.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "Attack_type" in categorical_cols:
    categorical_cols.remove("Attack_type")
df = pd.get_dummies(df, columns=categorical_cols)
feature_names = [col for col in df.columns if col != "Attack_type"]

# Preparar scaler
scaler = StandardScaler()
scaler.fit(df[feature_names])

# Treinar modelo RandomForestAnalysis (sem PCA para facilitar a demo)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Attack_type"] = le.fit_transform(df["Attack_type"])
X = df[feature_names]
y = df["Attack_type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf_analysis = RandomForestAnalysis(X_train_scaled, X_test_scaled, y_train, y_test, le, feature_names=feature_names, use_pca=False)
rf_analysis.model.fit(X_train_scaled, y_train)

# Frequência desejada dos ataques (ajuste conforme necessário)
attack_freq = [
    ("DOS_SYN_Hping", 94659),
    ("ARP_poisioning", 10064),
    ("Thing_Speak", 8467),
    ("MQTT_Publish", 4141),
    ("NMAP_XMAS_TREE_SCAN", 2006),
    ("NMAP_OS_DETECTION", 2000),
    ("NMAP_TCP_scan", 1004),
    ("DDOS_Slowloris", 540),
    ("Wipro_bulb", 203),
    ("Metasploit_Brute_Force_SSH", 27),
    ("NMAP_UDP_SCAN", 6)
]
attack_labels = [a[0] for a in attack_freq]
attack_weights = np.array([a[1] for a in attack_freq], dtype=float)
attack_weights = attack_weights / attack_weights.sum()

# Map attack label to encoded value
attack_label_to_code = {label: le.transform([label])[0] for label in attack_labels if label in le.classes_}

# Simular empresas
EMPRESAS = [
    "AlphaIoT", "BetaNetworks", "GammaTech", "DeltaSecure", "EpsilonCloud",
    "ZetaSystems", "ThetaCorp", "LambdaSolutions", "SigmaIoT", "OmegaNetworks"
]

# Histórico de ataques recebidos (agora armazena também as features enviadas)
attack_history = []

# Função para gerar dados random mais realistas usando amostragem dos dados reais
def generate_random_data():
    # Escolhe um ataque conforme a distribuição desejada
    chosen_attack = np.random.choice(attack_labels, p=attack_weights)
    # Seleciona uma linha real desse ataque
    attack_code = attack_label_to_code.get(chosen_attack)
    if attack_code is not None:
        row = df[df["Attack_type"] == attack_code].sample(1)
        random_row = row[feature_names].iloc[0].to_dict()
    else:
        # fallback: linha aleatória
        random_row = df[feature_names].sample(1).iloc[0].to_dict()
    # Adiciona campo empresa simulada
    empresa = random.choice(EMPRESAS)
    random_row['empresa'] = empresa
    return random_row

def simulate_attack_loop():
    while True:
        data = generate_random_data()
        X_input = pd.DataFrame([data])[feature_names]
        X_input_scaled = scaler.transform(X_input)
        pred = rf_analysis.model.predict(X_input_scaled)[0]
        pred_label = le.inverse_transform([pred])[0]
        empresa = data['empresa']
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        attack_history.append({
            "timestamp": timestamp,
            "empresa": empresa,
            "pred_label": pred_label,
            "is_attack": pred_label.lower() != "normal",
            "features": data
        })
        # Limita histórico para os últimos 100 ataques (para API), mas só mostra 10 na interface
        if len(attack_history) > 100:
            attack_history.pop(0)
        time.sleep(1.5)

# Inicia thread de simulação de ataques
threading.Thread(target=simulate_attack_loop, daemon=True).start()

# Flask app
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IoT Attack Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <meta charset="utf-8">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #14171f; margin: 0; padding: 0;}
        .container { max-width: 1200px; margin: 30px auto; background: #1e2230; border-radius: 18px; box-shadow: 0 4px 24px #000a; padding: 36px;}
        h1 { color: #fff; text-align: center; font-size: 2.8em; letter-spacing: 2px; margin-bottom: 10px;}
        .kpi-row { display: flex; justify-content: space-around; margin-bottom: 36px; gap: 24px;}
        .kpi { background: linear-gradient(135deg, #7c3aed 0%, #232936 100%); border-radius: 12px; padding: 28px 36px; text-align: center; box-shadow: 0 2px 8px #0003;}
        .kpi .value { font-size: 2.6em; font-weight: bold; color: #fff;}
        .kpi .label { font-size: 1.2em; color: #e0e6f0;}
        .charts-row { display: flex; gap: 30px; margin-bottom: 30px;}
        .chart-box { flex: 1; background: #232936; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px #0004;}
        h2 { color: #fff; margin-top: 30px; }
        .attack-list-container { max-height: 420px; overflow-y: auto; border-radius: 10px; background: #181c24; box-shadow: 0 1px 8px #0002; }
        .filter-row { display: flex; gap: 18px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; background: #232936; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 8px #0002;}
        .filter-label { color: #a78bfa; font-weight: bold; font-size: 1.08em; letter-spacing: 0.5px;}
        .filter-input, .filter-select { padding: 8px 14px; border-radius: 8px; border: 1px solid #7c3aed; background: #181c24; color: #fff; font-size: 1em; transition: border 0.2s, box-shadow 0.2s;}
        .filter-input:focus, .filter-select:focus { outline: none; border: 1.5px solid #a78bfa; box-shadow: 0 0 0 2px #7c3aed44;}
        .filter-select { min-width: 120px; }
        .history-table { width: 100%; border-collapse: collapse; }
        .history-table th, .history-table td { padding: 10px 14px; border-bottom: 1px solid #232936;}
        .history-table th { background: #232936; color: #b0b8c1; font-size: 1.1em;}
        .attack-row { color: #ff5c5c; font-weight: bold;}
        .normal-row { color: #27ae60; }
        .empresa-cell { font-weight: bold; }
        .details-btn { background: #7c3aed; color: #fff; border: none; border-radius: 5px; padding: 6px 18px; cursor: pointer; font-weight: bold; font-size: 1em; transition: background 0.2s;}
        .details-btn:hover { background: #a78bfa; }
        /* Modal */
        .modal { display: none; position: fixed; z-index: 999; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background: rgba(20,23,31,0.97);}
        .modal-content { background: #232936; margin: 5% auto; padding: 36px; border-radius: 14px; width: 90%; max-width: 700px; color: #f8fafc; box-shadow: 0 2px 12px #7c3aed88;}
        .close { color: #a78bfa; float: right; font-size: 32px; font-weight: bold; cursor: pointer; transition: color 0.2s;}
        .close:hover { color: #fff;}
        .details-table { width: 100%; border-collapse: collapse; margin-top: 18px;}
        .details-table th, .details-table td { padding: 8px 14px; border-bottom: 1px solid #2c3442;}
        .details-table th { background: #232936; color: #b0b8c1;}
        .empresa { color: #a78bfa; font-weight: bold; }
        /* Scrollbar dark */
        ::-webkit-scrollbar { width: 10px; background: #232936;}
        ::-webkit-scrollbar-thumb { background: #181c24; border-radius: 8px;}
        @media (max-width: 900px) {
            .charts-row { flex-direction: column; }
            .kpi-row { flex-direction: column; gap: 12px;}
            .filter-row { flex-direction: column; gap: 10px;}
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>IoT Attack Monitoring Dashboard</h1>
        <div class="kpi-row">
            <div class="kpi">
                <div class="value" id="kpi-total">-</div>
                <div class="label">Total Packets Received</div>
            </div>
            <div class="kpi">
                <div class="value" id="kpi-attacks">-</div>
                <div class="label">Detected Attacks</div>
            </div>
            <div class="kpi">
                <div class="value" id="kpi-empresas">-</div>
                <div class="label">Monitored Companies</div>
            </div>
            <div class="kpi">
                <div class="value" id="kpi-last">-</div>
                <div class="label">Last Attack</div>
            </div>
        </div>
        <div class="charts-row">
            <div class="chart-box">
                <div id="attack-bar"></div>
            </div>
            <div class="chart-box">
                <div id="empresa-pie"></div>
            </div>
        </div>
        <h2>Latest Received Attacks</h2>
        <div class="attack-list-container">
            <div class="filter-row">
                <span class="filter-label">Company:</span>
                <select id="filter-empresa" class="filter-select"><option value="">All</option></select>
                <span class="filter-label">Attack Type:</span>
                <select id="filter-tipo" class="filter-select"><option value="">All</option></select>
                <span class="filter-label">Status:</span>
                <select id="filter-status" class="filter-select">
                    <option value="">All</option>
                    <option value="Ataque">Attack</option>
                    <option value="Normal">Normal</option>
                </select>
                <span class="filter-label">Search:</span>
                <input id="filter-busca" class="filter-input" type="text" placeholder="Search by timestamp...">
            </div>
            <div id="history-table-div">
                {{history_table|safe}}
            </div>
        </div>
    </div>
    <!-- Modal for details -->
    <div id="details-modal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <h2 style="color:#a78bfa;">Attack Details</h2>
            <div id="details-content"></div>
        </div>
    </div>
    <script>
        let allRows = [];
        let lastFilters = {empresa: "", tipo: "", status: "", busca: ""};

        function updateDashboard() {
            fetch('/api/attack_stats')
            .then(r=>r.json())
            .then(function(data){
                if(document.getElementById('kpi-total')) document.getElementById('kpi-total').textContent = data.kpi_total;
                if(document.getElementById('kpi-attacks')) document.getElementById('kpi-attacks').textContent = data.kpi_attacks;
                if(document.getElementById('kpi-empresas')) document.getElementById('kpi-empresas').textContent = data.kpi_empresas;
                if(document.getElementById('kpi-last')) document.getElementById('kpi-last').textContent = data.kpi_last;
                Plotly.react('attack-bar', data.attack_bar.data, data.attack_bar.layout, {displayModeBar: false});
                Plotly.react('empresa-pie', data.empresa_pie.data, data.empresa_pie.layout, {displayModeBar: false});
                allRows = data.history_json;
                updateFilters();
                filterTable(true);
            });
        }
        function attachDetailsBtns() {
            document.querySelectorAll('.details-btn').forEach(function(btn){
                btn.onclick = function() {
                    var idx = btn.getAttribute('data-idx');
                    fetch('/api/attack_details/' + idx)
                    .then(r=>r.json())
                    .then(function(data){
                        var html = '<div class="empresa">Company: ' + (data.empresa || '') + '</div>';
                        html += '<table class="details-table"><tr><th>Variable</th><th>Value</th></tr>';
                        for (var k in data) {
                            if(k !== "empresa") html += '<tr><td>' + k + '</td><td>' + data[k] + '</td></tr>';
                        }
                        html += '</table>';
                        document.getElementById('details-content').innerHTML = html;
                        document.getElementById('details-modal').style.display = 'block';
                    });
                }
            });
        }
        function updateFilters() {
            let empresas = new Set(), tipos = new Set();
            allRows.forEach(r => {
                empresas.add(r.empresa);
                tipos.add(r.pred_label);
            });
            let empresaSel = document.getElementById('filter-empresa');
            let tipoSel = document.getElementById('filter-tipo');
            let empresaVal = lastFilters.empresa || empresaSel.value, tipoVal = lastFilters.tipo || tipoSel.value;
            empresaSel.innerHTML = '<option value="">All</option>' + Array.from(empresas).sort().map(e=>`<option value="${e}">${e}</option>`).join('');
            tipoSel.innerHTML = '<option value="">All</option>' + Array.from(tipos).sort().map(e=>`<option value="${e}">${e}</option>`).join('');
            empresaSel.value = empresaVal; tipoSel.value = tipoVal;
        }
        function filterTable(fromUpdateDashboard=false) {
            let empresa = fromUpdateDashboard ? lastFilters.empresa : document.getElementById('filter-empresa').value;
            let tipo = fromUpdateDashboard ? lastFilters.tipo : document.getElementById('filter-tipo').value;
            let status = fromUpdateDashboard ? lastFilters.status : document.getElementById('filter-status').value;
            let busca = fromUpdateDashboard ? lastFilters.busca : document.getElementById('filter-busca').value.toLowerCase();
            if(!fromUpdateDashboard) lastFilters = {empresa, tipo, status, busca};
            let filtered = allRows.filter(function(row){
                let ok = true;
                if(empresa && row.empresa !== empresa) ok = false;
                if(tipo && row.pred_label !== tipo) ok = false;
                if(status && ((status === "Ataque" && !row.is_attack) || (status === "Normal" && row.is_attack))) ok = false;
                if(busca && !row.timestamp.toLowerCase().includes(busca)) ok = false;
                return ok;
            });
            let table = `<table class="history-table">
                <tr>
                    <th>Timestamp</th>
                    <th>Company</th>
                    <th>Attack Type</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>`;
            filtered.forEach(function(row, idx){
                let row_class = row.is_attack ? "attack-row" : "normal-row";
                let status = row.is_attack ? "Attack" : "Normal";
                let real_idx = row.idx;
                table += `<tr class="${row_class}">
                    <td>${row.timestamp}</td>
                    <td class="empresa-cell">${row.empresa}</td>
                    <td>${row.pred_label}</td>
                    <td>${status}</td>
                    <td><button class="details-btn" data-idx="${real_idx}">View Details</button></td>
                </tr>`;
            });
            table += "</table>";
            document.getElementById('history-table-div').innerHTML = table;
            attachDetailsBtns();
        }
        document.addEventListener('change', function(e){
            if(e.target.classList.contains('filter-select')) filterTable();
        });
        document.addEventListener('input', function(e){
            if(e.target.classList.contains('filter-input')) filterTable();
        });
        updateDashboard();
        setInterval(updateDashboard, 1600);
        function closeModal() {
            document.getElementById('details-modal').style.display = 'none';
        }
        window.onclick = function(event) {
            var modal = document.getElementById('details-modal');
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }
    </script>
</body>
</html>
"""

def render_history_table(history):
    # Show up to 50 most recent attacks
    last_50 = list(reversed(history))[:50]
    table = """
    <table class="history-table">
        <tr>
            <th>Timestamp</th>
            <th>Company</th>
            <th>Attack Type</th>
            <th>Status</th>
            <th>Details</th>
        </tr>
    """
    for idx, row in enumerate(last_50):
        row_class = "attack-row" if row["is_attack"] else "normal-row"
        status = "Attack" if row["is_attack"] else "Normal"
        real_idx = len(history) - 1 - idx
        table += f"""
        <tr class="{row_class}">
            <td>{row['timestamp']}</td>
            <td class="empresa-cell">{row['empresa']}</td>
            <td>{row['pred_label']}</td>
            <td>{status}</td>
            <td><button class="details-btn" data-idx="{real_idx}">View Details</button></td>
        </tr>
        """
    table += "</table>"
    return table

@app.route("/")
def dashboard():
    history_table = render_history_table(attack_history)
    return render_template_string(
        HTML,
        history_table=history_table
    )

@app.route("/api/attack_stats")
def attack_stats():
    total = len(attack_history)
    attacks = sum(1 for x in attack_history if x["is_attack"])
    empresas = len(set(x["empresa"] for x in attack_history))
    last = attack_history[-1]["pred_label"] if attack_history else "-"
    labels = [x["pred_label"] for x in attack_history]
    if labels:
        attack_types, counts = np.unique(labels, return_counts=True)
    else:
        attack_types, counts = [], []
    attack_bar = dict(
        data=[dict(
            type="bar",
            x=attack_types.tolist(),
            y=counts.tolist(),
            marker=dict(color="#a78bfa")
        )],
        layout=dict(
            title="Frequência dos Tipos de Ataque",
            xaxis=dict(title="Tipo de Ataque"),
            yaxis=dict(title="Qtd"),
            height=300,
            paper_bgcolor="#232936",
            plot_bgcolor="#232936",
            font=dict(color="#f8fafc")
        )
    )
    empresas_list = [x["empresa"] for x in attack_history]
    if empresas_list:
        emp_types, emp_counts = np.unique(empresas_list, return_counts=True)
    else:
        emp_types, emp_counts = [], []
    empresa_pie = dict(
        data=[dict(
            type="pie",
            labels=emp_types.tolist(),
            values=emp_counts.tolist(),
            hole=0.4,
            marker=dict(colors=["#a78bfa", "#ff5c5c", "#27ae60", "#b0b8c1", "#f8fafc", "#232936", "#007bff", "#c0392b", "#181c24", "#2c3442"])
        )],
        layout=dict(
            title="Distribuição por Empresa",
            height=300,
            paper_bgcolor="#232936",
            font=dict(color="#f8fafc")
        )
    )
    # Para filtros, envie até 50 ataques recentes, com índice real
    last_50 = list(reversed(attack_history))[:50]
    history_json = []
    for idx, row in enumerate(last_50):
        real_idx = len(attack_history) - 1 - idx
        history_json.append({
            "timestamp": row["timestamp"],
            "empresa": row["empresa"],
            "pred_label": row["pred_label"],
            "is_attack": row["is_attack"],
            "idx": real_idx
        })
    history_table = render_history_table(attack_history)
    return jsonify({
        "kpi_total": total,
        "kpi_attacks": attacks,
        "kpi_empresas": empresas,
        "kpi_last": last,
        "attack_bar": attack_bar,
        "empresa_pie": empresa_pie,
        "history_table": history_table,
        "history_json": history_json
    })

@app.route("/api/attack_details/<int:idx>")
def attack_details(idx):
    if 0 <= idx < len(attack_history):
        features = attack_history[idx]["features"].copy()
        return jsonify(features)
    return jsonify({"erro": "Ataque não encontrado"}), 404

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
