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
attack_history_random = []

# Índice global para percorrer o dataset sequencialmente
current_dataset_index = 0

# Função para gerar dados sequenciais do dataset
def generate_sequential_data():
    global current_dataset_index
    
    # Se chegou ao fim do dataset, reinicia do início
    if current_dataset_index >= len(df):
        current_dataset_index = 0
    
    # Pega a linha atual do dataset
    row = df.iloc[current_dataset_index]
    sequential_row = row[feature_names].to_dict()
    
    # Adiciona campo empresa simulada
    empresa = random.choice(EMPRESAS)
    sequential_row['empresa'] = empresa
    
    # Incrementa o índice para a próxima iteração
    current_dataset_index += 1
    
    return sequential_row

# Função para gerar dados random (versão anterior)
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
        # Dados sequenciais
        data_seq = generate_sequential_data()
        X_input_seq = pd.DataFrame([data_seq])[feature_names]
        X_input_scaled_seq = scaler.transform(X_input_seq)
        pred_seq = rf_analysis.model.predict(X_input_scaled_seq)[0]
        pred_label_seq = le.inverse_transform([pred_seq])[0]
        empresa_seq = data_seq['empresa']
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        attack_history.append({
            "timestamp": timestamp,
            "empresa": empresa_seq,
            "pred_label": pred_label_seq,
            "is_attack": pred_label_seq.lower() != "normal",
            "features": data_seq
        })
        
        # Dados aleatórios
        data_rand = generate_random_data()
        X_input_rand = pd.DataFrame([data_rand])[feature_names]
        X_input_scaled_rand = scaler.transform(X_input_rand)
        pred_rand = rf_analysis.model.predict(X_input_scaled_rand)[0]
        pred_label_rand = le.inverse_transform([pred_rand])[0]
        empresa_rand = data_rand['empresa']
        
        attack_history_random.append({
            "timestamp": timestamp,
            "empresa": empresa_rand,
            "pred_label": pred_label_rand,
            "is_attack": pred_label_rand.lower() != "normal",
            "features": data_rand
        })
        
        # Limita histórico para os últimos 100 ataques
        if len(attack_history) > 100:
            attack_history.pop(0)
        if len(attack_history_random) > 100:
            attack_history_random.pop(0)
            
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
        
        /* Tabs */
        .tabs { display: flex; margin-bottom: 30px; border-bottom: 2px solid #232936;}
        .tab { background: #232936; color: #b0b8c1; padding: 12px 24px; cursor: pointer; border-radius: 8px 8px 0 0; margin-right: 4px; transition: all 0.3s;}
        .tab.active { background: #7c3aed; color: #fff; box-shadow: 0 -2px 8px #7c3aed44;}
        .tab:hover:not(.active) { background: #2c3442; color: #a78bfa;}
        .tab-content { display: none;}
        .tab-content.active { display: block;}
        
        .kpi-row { display: flex; justify-content: space-around; margin-bottom: 36px; gap: 24px;}
        .kpi { background: linear-gradient(135deg, #7c3aed 0%, #232936 100%); border-radius: 12px; padding: 28px 36px; text-align: center; box-shadow: 0 2px 8px #0003;}
        .kpi .value { font-size: 2.6em; font-weight: bold; color: #fff;}
        .kpi .label { font-size: 1.2em; color: #e0e6f0;}
        .charts-row { display: flex; gap: 30px; margin-bottom: 30px;}
        .chart-box { flex: 1; background: #232936; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px #0004; height: 340px; min-height: 340px; max-height: 340px;}
        .chart-box > div { height: 100% !important;}
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
            .tabs { flex-direction: column; gap: 4px;}
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>IoT Attack Monitoring Dashboard</h1>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('sequential')">Sequential Data</div>
            <div class="tab" onclick="switchTab('random')">Random Data</div>
        </div>
        
        <!-- Sequential Tab Content -->
        <div id="sequential-content" class="tab-content active">
            <div class="kpi-row">
                <div class="kpi">
                    <div class="value" id="kpi-total-seq">-</div>
                    <div class="label">Total Packets Received</div>
                </div>
                <div class="kpi">
                    <div class="value" id="kpi-attacks-seq">-</div>
                    <div class="label">Detected Attacks</div>
                </div>
                <div class="kpi">
                    <div class="value" id="kpi-empresas-seq">-</div>
                    <div class="label">Monitored Companies</div>
                </div>
                <div class="kpi">
                    <div class="value" id="kpi-last-seq">-</div>
                    <div class="label">Last Attack</div>
                </div>
            </div>
            <div class="charts-row">
                <div class="chart-box">
                    <div id="attack-bar-seq"></div>
                </div>
                <div class="chart-box">
                    <div id="empresa-pie-seq"></div>
                </div>
            </div>
            <h2>Latest Received Attacks (Sequential)</h2>
            <div class="attack-list-container">
                <div class="filter-row">
                    <span class="filter-label">Company:</span>
                    <select id="filter-empresa-seq" class="filter-select"><option value="">All</option></select>
                    <span class="filter-label">Attack Type:</span>
                    <select id="filter-tipo-seq" class="filter-select"><option value="">All</option></select>
                    <span class="filter-label">Status:</span>
                    <select id="filter-status-seq" class="filter-select">
                        <option value="">All</option>
                        <option value="Ataque">Attack</option>
                        <option value="Normal">Normal</option>
                    </select>
                    <span class="filter-label">Search:</span>
                    <input id="filter-busca-seq" class="filter-input" type="text" placeholder="Search by timestamp...">
                </div>
                <div id="history-table-div-seq">
                    {{history_table|safe}}
                </div>
            </div>
        </div>
        
        <!-- Random Tab Content -->
        <div id="random-content" class="tab-content">
            <div class="kpi-row">
                <div class="kpi">
                    <div class="value" id="kpi-total-rand">-</div>
                    <div class="label">Total Packets Received</div>
                </div>
                <div class="kpi">
                    <div class="value" id="kpi-attacks-rand">-</div>
                    <div class="label">Detected Attacks</div>
                </div>
                <div class="kpi">
                    <div class="value" id="kpi-empresas-rand">-</div>
                    <div class="label">Monitored Companies</div>
                </div>
                <div class="kpi">
                    <div class="value" id="kpi-last-rand">-</div>
                    <div class="label">Last Attack</div>
                </div>
            </div>
            <div class="charts-row">
                <div class="chart-box">
                    <div id="attack-bar-rand"></div>
                </div>
                <div class="chart-box">
                    <div id="empresa-pie-rand"></div>
                </div>
            </div>
            <h2>Latest Received Attacks (Random)</h2>
            <div class="attack-list-container">
                <div class="filter-row">
                    <span class="filter-label">Company:</span>
                    <select id="filter-empresa-rand" class="filter-select"><option value="">All</option></select>
                    <span class="filter-label">Attack Type:</span>
                    <select id="filter-tipo-rand" class="filter-select"><option value="">All</option></select>
                    <span class="filter-label">Status:</span>
                    <select id="filter-status-rand" class="filter-select">
                        <option value="">All</option>
                        <option value="Ataque">Attack</option>
                        <option value="Normal">Normal</option>
                    </select>
                    <span class="filter-label">Search:</span>
                    <input id="filter-busca-rand" class="filter-input" type="text" placeholder="Search by timestamp...">
                </div>
                <div id="history-table-div-rand">
                </div>
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
        let allRowsSeq = [], allRowsRand = [];
        let lastFiltersSeq = {empresa: "", tipo: "", status: "", busca: ""};
        let lastFiltersRand = {empresa: "", tipo: "", status: "", busca: ""};
        let currentTab = 'sequential';

        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName + '-content').classList.add('active');
            event.target.classList.add('active');
            currentTab = tabName;
            
            // Redimensionar gráficos após mudança de aba
            setTimeout(() => {
                if (tabName === 'sequential') {
                    Plotly.Plots.resize('attack-bar-seq');
                    Plotly.Plots.resize('empresa-pie-seq');
                } else if (tabName === 'random') {
                    Plotly.Plots.resize('attack-bar-rand');
                    Plotly.Plots.resize('empresa-pie-rand');
                }
            }, 100);
        }

        function updateDashboard() {
            // Update Sequential
            fetch('/api/attack_stats/sequential')
            .then(r=>r.json())
            .then(function(data){
                updateKPIs('seq', data);
                updateCharts('seq', data);
                allRowsSeq = data.history_json;
                updateFilters('seq');
                filterTable('seq', true);
            });
            
            // Update Random
            fetch('/api/attack_stats/random')
            .then(r=>r.json())
            .then(function(data){
                updateKPIs('rand', data);
                updateCharts('rand', data);
                allRowsRand = data.history_json;
                updateFilters('rand');
                filterTable('rand', true);
            });
        }
        
        function updateKPIs(suffix, data) {
            if(document.getElementById('kpi-total-' + suffix)) document.getElementById('kpi-total-' + suffix).textContent = data.kpi_total;
            if(document.getElementById('kpi-attacks-' + suffix)) document.getElementById('kpi-attacks-' + suffix).textContent = data.kpi_attacks;
            if(document.getElementById('kpi-empresas-' + suffix)) document.getElementById('kpi-empresas-' + suffix).textContent = data.kpi_empresas;
            if(document.getElementById('kpi-last-' + suffix)) document.getElementById('kpi-last-' + suffix).textContent = data.kpi_last;
        }
        
        function updateCharts(suffix, data) {
            // Layout com altura fixa
            const fixedLayout = {
                height: 300,
                width: null,
                autosize: true,
                paper_bgcolor: "#232936",
                plot_bgcolor: "#232936",
                font: {color: "#f8fafc"},
                margin: { l: 40, r: 40, t: 40, b: 40 }
            };
            
            // Atualizar layout do gráfico de barras
            const barLayout = Object.assign({}, data.attack_bar.layout, fixedLayout, {
                title: "Frequência dos Tipos de Ataque",
                xaxis: {title: "Tipo de Ataque", color: "#f8fafc"},
                yaxis: {title: "Qtd", color: "#f8fafc"}
            });
            
            // Atualizar layout do gráfico de pizza
            const pieLayout = Object.assign({}, data.empresa_pie.layout, fixedLayout, {
                title: "Distribuição por Empresa"
            });
            
            const config = {
                displayModeBar: false, 
                responsive: true,
                staticPlot: false
            };
            
            Plotly.react('attack-bar-' + suffix, data.attack_bar.data, barLayout, config);
            Plotly.react('empresa-pie-' + suffix, data.empresa_pie.data, pieLayout, config);
        }

        function attachDetailsBtns(suffix) {
            document.querySelectorAll('.details-btn').forEach(function(btn){
                btn.onclick = function() {
                    var idx = btn.getAttribute('data-idx');
                    var type = btn.getAttribute('data-type') || 'sequential';
                    fetch('/api/attack_details/' + type + '/' + idx)
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
        
        function updateFilters(suffix) {
            let allRows = suffix === 'seq' ? allRowsSeq : allRowsRand;
            let empresas = new Set(), tipos = new Set();
            allRows.forEach(r => {
                empresas.add(r.empresa);
                tipos.add(r.pred_label);
            });
            let empresaSel = document.getElementById('filter-empresa-' + suffix);
            let tipoSel = document.getElementById('filter-tipo-' + suffix);
            let lastFilters = suffix === 'seq' ? lastFiltersSeq : lastFiltersRand;
            let empresaVal = lastFilters.empresa || empresaSel.value, tipoVal = lastFilters.tipo || tipoSel.value;
            empresaSel.innerHTML = '<option value="">All</option>' + Array.from(empresas).sort().map(e=>`<option value="${e}">${e}</option>`).join('');
            tipoSel.innerHTML = '<option value="">All</option>' + Array.from(tipos).sort().map(e=>`<option value="${e}">${e}</option>`).join('');
            empresaSel.value = empresaVal; tipoSel.value = tipoVal;
        }
        
        function filterTable(suffix, fromUpdateDashboard=false) {
            let lastFilters = suffix === 'seq' ? lastFiltersSeq : lastFiltersRand;
            let allRows = suffix === 'seq' ? allRowsSeq : allRowsRand;
            let empresa = fromUpdateDashboard ? lastFilters.empresa : document.getElementById('filter-empresa-' + suffix).value;
            let tipo = fromUpdateDashboard ? lastFilters.tipo : document.getElementById('filter-tipo-' + suffix).value;
            let status = fromUpdateDashboard ? lastFilters.status : document.getElementById('filter-status-' + suffix).value;
            let busca = fromUpdateDashboard ? lastFilters.busca : document.getElementById('filter-busca-' + suffix).value.toLowerCase();
            if(!fromUpdateDashboard) {
                if(suffix === 'seq') lastFiltersSeq = {empresa, tipo, status, busca};
                else lastFiltersRand = {empresa, tipo, status, busca};
            }
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
                let dataType = suffix === 'seq' ? 'sequential' : 'random';
                table += `<tr class="${row_class}">
                    <td>${row.timestamp}</td>
                    <td class="empresa-cell">${row.empresa}</td>
                    <td>${row.pred_label}</td>
                    <td>${status}</td>
                    <td><button class="details-btn" data-idx="${real_idx}" data-type="${dataType}">View Details</button></td>
                </tr>`;
            });
            table += "</table>";
            document.getElementById('history-table-div-' + suffix).innerHTML = table;
            attachDetailsBtns(suffix);
        }
        
        document.addEventListener('change', function(e){
            if(e.target.classList.contains('filter-select')) {
                let suffix = e.target.id.includes('-seq') ? 'seq' : 'rand';
                filterTable(suffix);
            }
        });
        document.addEventListener('input', function(e){
            if(e.target.classList.contains('filter-input')) {
                let suffix = e.target.id.includes('-seq') ? 'seq' : 'rand';
                filterTable(suffix);
            }
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

@app.route("/api/attack_stats/sequential")
def attack_stats_sequential():
    return get_attack_stats(attack_history)

@app.route("/api/attack_stats/random")
def attack_stats_random():
    return get_attack_stats(attack_history_random)

def get_attack_stats(history):
    total = len(history)
    attacks = sum(1 for x in history if x["is_attack"])
    empresas = len(set(x["empresa"] for x in history))
    last = history[-1]["pred_label"] if history else "-"
    labels = [x["pred_label"] for x in history]
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
    empresas_list = [x["empresa"] for x in history]
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
    last_50 = list(reversed(history))[:50]
    history_json = []
    for idx, row in enumerate(last_50):
        real_idx = len(history) - 1 - idx
        history_json.append({
            "timestamp": row["timestamp"],
            "empresa": row["empresa"],
            "pred_label": row["pred_label"],
            "is_attack": row["is_attack"],
            "idx": real_idx
        })
    return jsonify({
        "kpi_total": total,
        "kpi_attacks": attacks,
        "kpi_empresas": empresas,
        "kpi_last": last,
        "attack_bar": attack_bar,
        "empresa_pie": empresa_pie,
        "history_json": history_json
    })

@app.route("/api/attack_details/sequential/<int:idx>")
def attack_details_sequential(idx):
    if 0 <= idx < len(attack_history):
        features = attack_history[idx]["features"].copy()
        return jsonify(features)
    return jsonify({"erro": "Ataque não encontrado"}), 404

@app.route("/api/attack_details/random/<int:idx>")
def attack_details_random(idx):
    if 0 <= idx < len(attack_history_random):
        features = attack_history_random[idx]["features"].copy()
        return jsonify(features)
    return jsonify({"erro": "Ataque não encontrado"}), 404

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
