import pandas as pd
import numpy as np
from flask import Flask, render_template_string, jsonify, request
from sklearn.preprocessing import StandardScaler
from RandomForestAnalysis import RandomForestAnalysis
import random
import threading
import time
import plotly
import plotly.graph_objs as go
import json
import os

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

# --- Mantém apenas a simulação random para a aba Random ---
attack_history_random = []

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
        # Dados aleatórios
        data_rand = generate_random_data()
        X_input_rand = pd.DataFrame([data_rand])[feature_names]
        X_input_scaled_rand = scaler.transform(X_input_rand)
        pred_rand = rf_analysis.model.predict(X_input_scaled_rand)[0]
        pred_label_rand = le.inverse_transform([pred_rand])[0]
        empresa_rand = data_rand['empresa']
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        attack_history_random.append({
            "timestamp": timestamp,
            "empresa": empresa_rand,
            "pred_label": pred_label_rand,
            "is_attack": pred_label_rand.lower() != "normal",
            "features": data_rand
        })
        # Remove the limit - let it grow unlimited
        # if len(attack_history_random) > 100:
        #     attack_history_random.pop(0)
        time.sleep(1.5)

threading.Thread(target=simulate_attack_loop, daemon=True).start()

# --- CSV upload e análise para a aba Sequential ---
UPLOAD_FOLDER = "uploaded_csvs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
uploaded_csv_data = {"df": None, "filename": None}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
        .chart-box { 
            flex: 1; 
            background: linear-gradient(135deg, #232936 0%, #2c3442 100%); 
            border-radius: 15px; 
            padding: 25px; 
            box-shadow: 0 8px 25px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.05); 
            height: 360px; 
            min-height: 360px; 
            max-height: 360px;
            position: relative;
            overflow: hidden;
        }
        .chart-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(124,58,237,0.5), transparent);
        }
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
        .upload-box { background: #232936; border-radius: 12px; padding: 28px 36px; margin-bottom: 30px; text-align: center; box-shadow: 0 2px 8px #0003;}
        .upload-label { color: #a78bfa; font-size: 1.2em; font-weight: bold; margin-bottom: 10px; display: block;}
        .upload-input { padding: 10px 18px; border-radius: 8px; border: 1.5px solid #7c3aed; background: #181c24; color: #fff; font-size: 1em; margin-bottom: 10px;}
        .upload-btn { background: #7c3aed; color: #fff; border: none; border-radius: 8px; padding: 10px 28px; cursor: pointer; font-weight: bold; font-size: 1.1em; transition: background 0.2s;}
        .upload-btn:hover { background: #a78bfa; }
        .csv-info { color: #b0b8c1; margin-bottom: 10px; }
        .sortable { cursor: pointer; }
        .sort-arrow { font-size: 0.9em; margin-left: 4px; }
        .csv-table-container { max-height: 420px; overflow-y: auto; border-radius: 10px; background: #181c24; box-shadow: 0 1px 8px #0002; margin-top: 18px;}
        .csv-table { width: 100%; border-collapse: collapse; }
        .csv-table th, .csv-table td { padding: 10px 14px; border-bottom: 1px solid #232936;}
        .csv-table th { background: #232936; color: #b0b8c1; font-size: 1.1em;}
        .csv-table tr:hover { background: #23293655; }
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
            <div class="tab active" onclick="switchTab('sequential')">Sequential Data (CSV Analysis)</div>
            <div class="tab" onclick="switchTab('random')">Random Data</div>
        </div>
        <!-- Sequential Tab Content -->
        <div id="sequential-content" class="tab-content active">
            <div class="upload-box">
                <form id="csv-upload-form" enctype="multipart/form-data">
                    <label class="upload-label">Upload CSV file (e.g., resultados_com_predicoes.csv):</label>
                    <input class="upload-input" type="file" name="csvfile" accept=".csv" required>
                    <button class="upload-btn" type="submit">Upload & Analyze</button>
                </form>
                <div id="csv-info" class="csv-info"></div>
            </div>
            <div id="csv-dashboard" style="display:none;">
                <div class="kpi-row" id="csv-kpis"></div>
                <div class="charts-row">
                    <div class="chart-box">
                        <div id="csv-attack-bar"></div>
                    </div>
                    <div class="chart-box">
                        <div id="csv-attack-pie"></div>
                    </div>
                </div>
                <h2>Attack Type Summary Table</h2>
                <div class="csv-table-container">
                    <div id="csv-table-div"></div>
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
        // --- CSV Upload & Dashboard ---
        let csvTableData = [];
        let csvSortCol = "count";
        let csvSortDir = "desc";
        document.getElementById('csv-upload-form').onsubmit = function(e){
            e.preventDefault();
            let formData = new FormData(this);
            document.getElementById('csv-info').textContent = "Uploading and analyzing...";
            fetch('/api/upload_csv', {method:'POST', body:formData})
            .then(r=>r.json())
            .then(function(data){
                if(data.status === "ok") {
                    document.getElementById('csv-info').textContent = "File: " + data.filename + " (" + data.n_rows + " rows)";
                    showCsvDashboard(data);
                } else {
                    document.getElementById('csv-info').textContent = "Error: " + data.error;
                    document.getElementById('csv-dashboard').style.display = "none";
                }
            });
        };
        function showCsvDashboard(data) {
            document.getElementById('csv-dashboard').style.display = "";
            // KPIs
            let kpiHtml = `
                <div class="kpi">
                    <div class="value">${data.n_rows}</div>
                    <div class="label">Total Packets</div>
                </div>
                <div class="kpi">
                    <div class="value">${data.n_attacks}</div>
                    <div class="label">Detected Attacks</div>
                </div>
                <div class="kpi">
                    <div class="value">${data.n_normals}</div>
                    <div class="label">Normal Packets</div>
                </div>
                <div class="kpi">
                    <div class="value">${data.n_types}</div>
                    <div class="label">Attack Types</div>
                </div>
            `;
            document.getElementById('csv-kpis').innerHTML = kpiHtml;
            // Charts
            Plotly.react('csv-attack-bar', data.attack_bar.data, data.attack_bar.layout, {displayModeBar:false, responsive:true});
            Plotly.react('csv-attack-pie', data.attack_pie.data, data.attack_pie.layout, {displayModeBar:false, responsive:true});
            // Table
            csvTableData = data.table_data;
            csvSortCol = "count";
            csvSortDir = "desc";
            renderCsvTable();
        }
        function renderCsvTable() {
            let sorted = [...csvTableData];
            sorted.sort((a,b)=>{
                if(csvSortCol==="count") return csvSortDir==="desc"?b.count-a.count:a.count-b.count;
                if(csvSortCol==="percent") return csvSortDir==="desc"?b.percent-a.percent:a.percent-b.percent;
                if(csvSortCol==="type") return csvSortDir==="desc"?b.type.localeCompare(a.type):a.type.localeCompare(b.type);
                return 0;
            });
            let th = (col, label) => `<th class="sortable" onclick="sortCsvTable('${col}')">${label}${csvSortCol===col?'<span class="sort-arrow">'+(csvSortDir==="desc"?"&#8595;":"&#8593;")+'</span>':''}</th>`;
            let html = `<table class="csv-table">
                <tr>
                    ${th('type','Attack Type')}
                    ${th('count','Count')}
                    ${th('percent','Percent')}
                </tr>`;
            sorted.forEach(row=>{
                let row_class = row.type.toLowerCase()==="normal" ? "normal-row" : "attack-row";
                html += `<tr class="${row_class}">
                    <td>${row.type}</td>
                    <td>${row.count}</td>
                    <td>${row.percent.toFixed(2)}%</td>
                </tr>`;
            });
            html += "</table>";
            document.getElementById('csv-table-div').innerHTML = html;
        }
        window.sortCsvTable = function(col) {
            if(csvSortCol===col) csvSortDir = csvSortDir==="desc"?"asc":"desc";
            else { csvSortCol=col; csvSortDir="desc"; }
            renderCsvTable();
        }
        // --- Tabs and random tab logic (unchanged) ---
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
            // Only update Random tab automatically
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
            const config = {
                displayModeBar: false, 
                responsive: true,
                staticPlot: false,
                modeBarButtonsToRemove: ['pan2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d','resetScale2d'],
                doubleClick: false
            };
            
            // Usar os dados com layout já otimizado do backend
            Plotly.react('attack-bar-' + suffix, data.attack_bar.data, data.attack_bar.layout, config);
            Plotly.react('empresa-pie-' + suffix, data.empresa_pie.data, data.empresa_pie.layout, config);
            
            // Adicionar animação suave
            setTimeout(() => {
                if (document.getElementById('attack-bar-' + suffix)) {
                    Plotly.animate('attack-bar-' + suffix, {
                        data: data.attack_bar.data,
                        layout: data.attack_bar.layout
                    }, {
                        transition: { duration: 500, easing: 'cubic-in-out' },
                        frame: { duration: 500 }
                    });
                }
            }, 100);
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

# --- API para upload e análise do CSV ---
@app.route("/api/upload_csv", methods=["POST"])
def upload_csv():
    if "csvfile" not in request.files:
        return jsonify({"status": "error", "error": "No file part"})
    file = request.files["csvfile"]
    if file.filename == "":
        return jsonify({"status": "error", "error": "No selected file"})
    try:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        df_csv = pd.read_csv(filepath)
        uploaded_csv_data["df"] = df_csv
        uploaded_csv_data["filename"] = filename
        # --- Análise dos dados agrupando ataques por nome correto ---
        col_pred = None
        for c in df_csv.columns:
            if "pred" in c.lower() or "attack_type" in c.lower():
                col_pred = c
                break
        if col_pred is None:
            return jsonify({"status":"error", "error":"No prediction/attack_type column found"})
        total = len(df_csv)
        # Se a coluna for numérica, tenta decodificar usando o LabelEncoder do modelo principal
        col_values = df_csv[col_pred]
        # Detecta se é numérica (int/float) e converte para string se necessário
        if np.issubdtype(col_values.dtype, np.number):
            # Usa o label encoder treinado no início do script
            try:
                decoded = le.inverse_transform(col_values.astype(int))
                col_values = pd.Series(decoded, index=col_values.index)
            except Exception:
                # fallback: mostra como está
                col_values = col_values.astype(str)
        # Agrupa todos os ataques diferentes de "Normal" em "Attack"
        grouped = col_values.apply(lambda x: "Normal" if str(x).lower() == "normal" else str(x))
        type_counts = grouped.value_counts(dropna=False)
        n_types = type_counts.shape[0]
        n_attacks = type_counts.drop(labels=["Normal"], errors="ignore").sum() if "Normal" in type_counts else total-type_counts.get("Normal",0)
        n_normals = type_counts.get("Normal", 0)
        table_data = []
        for t, cnt in type_counts.items():
            percent = 100.0 * cnt / total if total else 0
            table_data.append({"type": str(t), "count": int(cnt), "percent": percent})
        # Gráfico de barras
        bar_colors = [
            "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57",
            "#ff9ff3", "#54a0ff", "#5f27cd", "#00d2d3", "#ff9f43",
            "#a55eea", "#26de81", "#fd79a8", "#6c5ce7", "#fdcb6e"
        ]
        attack_bar = dict(
            data=[dict(
                type="bar",
                x=[x["type"] for x in table_data],
                y=[x["count"] for x in table_data],
                marker=dict(
                    color=bar_colors[:len(table_data)],
                    line=dict(color='rgba(255,255,255,0.1)', width=1),
                    opacity=0.9
                ),
                text=[x["count"] for x in table_data],
                textposition='auto',
                textfont=dict(color='white', size=12, family='Segoe UI')
            )],
            layout=dict(
                title=dict(
                    text="Attack Type Frequency",
                    font=dict(size=18, color="#ffffff", family='Segoe UI'),
                    x=0.5
                ),
                xaxis=dict(
                    title=dict(text="Attack Type", font=dict(size=14, color="#e0e6f0")),
                    tickfont=dict(size=10, color="#b0b8c1"),
                    gridcolor="rgba(255,255,255,0.1)",
                    showgrid=True
                ),
                yaxis=dict(
                    title=dict(text="Count", font=dict(size=14, color="#e0e6f0")),
                    tickfont=dict(size=12, color="#b0b8c1"),
                    gridcolor="rgba(255,255,255,0.1)",
                    showgrid=True
                ),
                height=300,
                paper_bgcolor="#232936",
                plot_bgcolor="rgba(35,41,54,0.8)",
                font=dict(color="#f8fafc", family='Segoe UI'),
                margin=dict(l=50, r=30, t=50, b=80),
                showlegend=False,
                hovermode='closest'
            )
        )
        # Gráfico treemap - substituído o gráfico de barras horizontal
        pie_colors = [
            "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57",
            "#ff9ff3", "#54a0ff", "#5f27cd", "#00d2d3", "#ff9f43",
            "#a55eea", "#26de81", "#fd79a8", "#6c5ce7", "#fdcb6e"
        ]
        
        # Ordenar dados por contagem para melhor visualização
        sorted_data = sorted(table_data, key=lambda x: x["count"], reverse=True)
        
        attack_pie = dict(
            data=[dict(
                type="treemap",
                labels=[x["type"] for x in sorted_data],
                values=[x["count"] for x in sorted_data],
                parents=[""] * len(sorted_data),
                textinfo="label+value+percent parent",
                textfont=dict(size=12, color='white', family='Segoe UI'),
                marker=dict(
                    colors=pie_colors[:len(sorted_data)],
                    line=dict(color='rgba(255,255,255,0.3)', width=2),
                    colorscale='Viridis'
                ),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percentParent}<extra></extra>',
                pathbar=dict(visible=False),
                maxdepth=1
            )],
            layout=dict(
                title=dict(
                    text="Attack Type Distribution",
                    font=dict(size=18, color="#ffffff", family='Segoe UI'),
                    x=0.5
                ),
                height=300,
                paper_bgcolor="#232936",
                plot_bgcolor="rgba(35,41,54,0.8)",
                font=dict(color="#f8fafc", family='Segoe UI'),
                margin=dict(l=20, r=20, t=50, b=20),
                hoverlabel=dict(
                    bgcolor="rgba(35,41,54,0.9)",
                    bordercolor="rgba(255,255,255,0.2)",
                    font=dict(color="white", size=12)
                )
            )
        )
        return jsonify({
            "status": "ok",
            "filename": filename,
            "n_rows": total,
            "n_types": n_types,
            "n_attacks": int(n_attacks),
            "n_normals": int(n_normals),
            "table_data": table_data,
            "attack_bar": attack_bar,
            "attack_pie": attack_pie
        })
    except Exception as ex:
        return jsonify({"status": "error", "error": str(ex)})

# --- Dashboard principal ---
@app.route("/")
def dashboard():
    return render_template_string(
        HTML,
        history_table=""  # não mostra tabela na aba sequential
    )

# --- Random tab APIs (inalteradas) ---
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
    
    # Cores gradientes para o gráfico de barras
    bar_colors = [
        "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57",
        "#ff9ff3", "#54a0ff", "#5f27cd", "#00d2d3", "#ff9f43",
        "#a55eea", "#26de81", "#fd79a8", "#6c5ce7", "#fdcb6e"
    ]
    
    attack_bar = dict(
        data=[dict(
            type="bar",
            x=attack_types.tolist(),
            y=counts.tolist(),
            marker=dict(
                color=bar_colors[:len(attack_types)],
                line=dict(color='rgba(255,255,255,0.1)', width=1),
                opacity=0.9
            ),
            text=counts.tolist(),
            textposition='auto',
            textfont=dict(color='white', size=12, family='Segoe UI')
        )],
        layout=dict(
            title=dict(
                text="Frequência dos Tipos de Ataque",
                font=dict(size=18, color="#ffffff", family='Segoe UI'),
                x=0.5
            ),
            xaxis=dict(
                title=dict(text="Tipo de Ataque", font=dict(size=14, color="#e0e6f0")),
                tickfont=dict(size=10, color="#b0b8c1"),
                gridcolor="rgba(255,255,255,0.1)",
                showgrid=True
            ),
            yaxis=dict(
                title=dict(text="Quantidade", font=dict(size=14, color="#e0e6f0")),
                tickfont=dict(size=12, color="#b0b8c1"),
                gridcolor="rgba(255,255,255,0.1)",
                showgrid=True
            ),
            height=300,
            paper_bgcolor="#232936",
            plot_bgcolor="rgba(35,41,54,0.8)",
            font=dict(color="#f8fafc", family='Segoe UI'),
            margin=dict(l=50, r=30, t=50, b=80),
            showlegend=False,
            hovermode='closest'
        )
    )
    
    empresas_list = [x["empresa"] for x in history]
    if empresas_list:
        emp_types, emp_counts = np.unique(empresas_list, return_counts=True)
    else:
        emp_types, emp_counts = [], []
    
    # Cores vibrantes para o gráfico de pizza
    pie_colors = [
        "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57",
        "#ff9ff3", "#54a0ff", "#5f27cd", "#00d2d3", "#ff9f43",
        "#a55eea", "#26de81", "#fd79a8", "#6c5ce7", "#fdcb6e"
    ]
    
    empresa_pie = dict(
        data=[dict(
            type="sunburst",
            labels=emp_types.tolist(),
            values=emp_counts.tolist(),
            parents=[""] * len(emp_types),
            marker=dict(
                colors=pie_colors[:len(emp_types)],
                line=dict(color='rgba(255,255,255,0.2)', width=2)
            ),
            textfont=dict(size=11, color='white', family='Segoe UI'),
            textinfo='label+percent parent',
            hovertemplate='<b>%{label}</b><br>Ataques: %{value}<br>Percentual: %{percentParent}<extra></extra>',
            maxdepth=1,
            branchvalues="total"
        )],
        layout=dict(
            title=dict(
                text="Distribuição por Empresa",
                font=dict(size=18, color="#ffffff", family='Segoe UI'),
                x=0.5
            ),
            height=300,
            paper_bgcolor="#232936",
            plot_bgcolor="rgba(35,41,54,0.8)",
            font=dict(color="#f8fafc", family='Segoe UI'),
            margin=dict(l=20, r=20, t=50, b=20),
            hoverlabel=dict(
                bgcolor="rgba(35,41,54,0.9)",
                bordercolor="rgba(255,255,255,0.2)",
                font=dict(color="white", size=12)
            )
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
        });
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
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
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
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
