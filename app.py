import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.figure_factory as ff
from prophet import Prophet
import pmdarima as pm
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from jinja2 import Template
import base64
from pathlib import Path
import os
from scipy import stats
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(
    page_title="E-Ticaret Satış Analizi",
    page_icon="📊",
    layout="wide"
)

# Yardımcı fonksiyonlar
def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Veri çerçevesinin gerekli sütunları içerip içermediğini kontrol eder."""
    required_columns = ['tarih', 'urun_adi', 'miktar', 'satis_tutari']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Eksik sütunlar: {', '.join(missing_columns)}"
    return True, "Veri doğrulama başarılı"

def detect_and_convert_date(df: pd.DataFrame, date_column: str = 'tarih') -> pd.DataFrame:
    """Tarih sütununu otomatik olarak tespit edip dönüştürür."""
    if date_column in df.columns:
        try:
            # Farklı tarih formatlarını dene
            date_formats = [
                '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
                '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
                '%d.%m.%Y', '%Y.%m.%d'
            ]
            
            for date_format in date_formats:
                try:
                    df[date_column] = pd.to_datetime(df[date_column], format=date_format)
                    break
                except:
                    continue
            
            # Eğer yukarıdaki formatlar çalışmazsa, pandas'ın otomatik dönüşümünü dene
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column])
                
        except Exception as e:
            st.error(f"Tarih dönüşümünde hata: {str(e)}")
    
    return df

def load_data(uploaded_file: Any) -> Optional[pd.DataFrame]:
    """Farklı formatlardaki dosyaları yükler ve DataFrame'e dönüştürür."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Desteklenmeyen dosya formatı!")
            return None
            
        return df
    except Exception as e:
        st.error(f"Dosya yükleme hatası: {str(e)}")
        return None

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """RFM analizi yapar ve müşteri segmentasyonu oluşturur."""
    # Son işlem tarihini bul
    max_date = df['tarih'].max()
    
    # Sipariş ID kontrolü - eğer yoksa tarih sütunuyla sipariş sayısı hesaplayalım
    if 'siparis_id' not in df.columns:
        # Gruplandırarak bir proxy sipariş ID oluşturalım
        df['siparis_id_proxy'] = df.groupby(['musteri_id', df['tarih'].dt.date]).ngroup()
        frequency_col = 'siparis_id_proxy'
    else:
        frequency_col = 'siparis_id'
    
    # RFM metriklerini hesapla
    rfm = df.groupby('musteri_id').agg({
        'tarih': lambda x: (max_date - x.max()).days,  # Recency
        frequency_col: 'count',  # Frequency
        'satis_tutari': 'sum'  # Monetary
    }).rename(columns={
        'tarih': 'recency',
        frequency_col: 'frequency',
        'satis_tutari': 'monetary'
    })
    
    # RFM skorlarını hesapla (1-5 arası)
    
    # Recency için qcut kullan (küçük değerler daha iyi)
    r_labels = range(5, 0, -1)
    # Eğer 5'ten az farklı değer varsa, duplicates='drop' soruna neden olabilir
    if len(rfm['recency'].unique()) < 5:
        # Basit bir manual dönüşüm yapalım
        rfm['R'] = pd.cut(rfm['recency'], 
                        bins=[0, rfm['recency'].quantile(0.2), rfm['recency'].quantile(0.4), 
                              rfm['recency'].quantile(0.6), rfm['recency'].quantile(0.8), float('inf')], 
                        labels=[5, 4, 3, 2, 1], 
                        include_lowest=True)
    else:
        # Yeterli veri varsa qcut kullanalım
        try:
            r_quartiles = pd.qcut(rfm['recency'], q=5, labels=r_labels, duplicates='drop')
            rfm['R'] = r_quartiles
        except ValueError:
            # Yine de hata alırsak manual dönüşüm yapalım
            rfm['R'] = pd.cut(rfm['recency'],
                            bins=[0, rfm['recency'].quantile(0.2), rfm['recency'].quantile(0.4),
                                rfm['recency'].quantile(0.6), rfm['recency'].quantile(0.8), float('inf')],
                            labels=[5, 4, 3, 2, 1],
                            include_lowest=True)
    
    # Frequency için manuel skorlama
    def score_frequency(x):
        if x <= 1:
            return 1
        elif x <= 2:
            return 2
        elif x <= 3:
            return 3
        elif x <= 5:
            return 4
        else:
            return 5
    
    # Monetary için manuel skorlama
    def score_monetary(x, percentiles):
        if x <= percentiles[0]:
            return 1
        elif x <= percentiles[1]:
            return 2
        elif x <= percentiles[2]:
            return 3
        elif x <= percentiles[3]:
            return 4
        else:
            return 5
    
    # Monetary için yüzdelik dilimler
    monetary_percentiles = np.percentile(rfm['monetary'], [20, 40, 60, 80])
    
    # Skorları hesapla
    rfm['F'] = rfm['frequency'].apply(score_frequency)
    rfm['M'] = rfm['monetary'].apply(lambda x: score_monetary(x, monetary_percentiles))
    
    # RFM skorunu hesapla
    rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    
    # Müşteri segmentlerini belirle
    def segment_customers(row: pd.Series) -> str:
        if row['R'] >= 4 and row['F'] >= 4 and row['M'] >= 4:
            return 'VIP Müşteriler'
        elif row['R'] >= 3 and row['F'] >= 3 and row['M'] >= 3:
            return 'Sadık Müşteriler'
        elif row['R'] >= 2 and row['F'] >= 2 and row['M'] >= 2:
            return 'Potansiyel Müşteriler'
        else:
            return 'Risk Altındaki Müşteriler'
    
    rfm['Segment'] = rfm.apply(segment_customers, axis=1)
    
    return rfm

def analyze_time_series(df: pd.DataFrame) -> Dict[str, Any]:
    """Zaman serisi analizi yapar."""
    # Günlük satışları hesapla
    daily_sales = df.groupby('tarih')['satis_tutari'].sum().reset_index()
    daily_sales.set_index('tarih', inplace=True)
    
    # Hareketli ortalamalar
    daily_sales['MA7'] = daily_sales['satis_tutari'].rolling(window=7).mean()
    daily_sales['MA30'] = daily_sales['satis_tutari'].rolling(window=30).mean()
    
    # Mevsimsellik analizi
    try:
        decomposition = seasonal_decompose(daily_sales['satis_tutari'], period=30)
        seasonal = decomposition.seasonal
        trend = decomposition.trend
        residual = decomposition.resid
    except:
        seasonal = trend = residual = None
    
    # Büyüme oranları
    daily_sales['growth_rate'] = daily_sales['satis_tutari'].pct_change() * 100
    
    return {
        'daily_sales': daily_sales,
        'seasonal': seasonal,
        'trend': trend,
        'residual': residual
    }

def create_sales_heatmap(df: pd.DataFrame) -> go.Figure:
    """Günlük satış yoğunluğu için ısı haritası oluşturur."""
    # Günlük satışları hesapla
    df['satis_gunu'] = df['tarih'].dt.date
    df['satis_saati'] = df['tarih'].dt.hour
    
    daily_sales = df.groupby(['satis_gunu', 'satis_saati'])['satis_tutari'].sum().reset_index()
    
    # Pivot tablo oluştur
    pivot_table = daily_sales.pivot_table(
        values='satis_tutari',
        index='satis_saati',
        columns='satis_gunu',
        aggfunc='sum'
    )
    
    # Isı haritası oluştur
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Günlük Satış Yoğunluğu',
        xaxis_title='Tarih',
        yaxis_title='Saat',
        height=500
    )
    
    return fig

def analyze_categories(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Kategori bazlı analiz yapar."""
    if 'kategori' not in df.columns:
        return None
    
    # Kategori bazlı metrikler
    category_metrics = df.groupby('kategori').agg({
        'satis_tutari': ['sum', 'mean', 'count'],
        'miktar': 'sum'
    }).round(2)
    
    category_metrics.columns = ['Toplam Satış', 'Ortalama Satış', 'Sipariş Sayısı', 'Toplam Miktar']
    
    # Kategori büyüme oranları
    category_growth = df.groupby(['kategori', pd.Grouper(key='tarih', freq='M')])['satis_tutari'].sum().reset_index()
    category_growth['growth'] = category_growth.groupby('kategori')['satis_tutari'].pct_change() * 100
    
    return {
        'metrics': category_metrics,
        'growth': category_growth
    }

def forecast_sales(df: pd.DataFrame, forecast_days: int = 30) -> Dict[str, Any]:
    """Satış tahmini yapar."""
    # Günlük satışları hesapla
    daily_sales = df.groupby('tarih')['satis_tutari'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    
    # Prophet modeli
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(daily_sales)
    
    # Gelecek tarihleri oluştur
    future_dates = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future_dates)
    
    # ARIMA modeli
    arima_model = pm.auto_arima(daily_sales['y'],
                               seasonal=True,
                               m=7,
                               suppress_warnings=True)
    
    arima_forecast = arima_model.predict(n_periods=forecast_days)
    
    return {
        'prophet_forecast': forecast,
        'arima_forecast': arima_forecast,
        'last_date': daily_sales['ds'].max()
    }

def optimize_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Stok optimizasyonu önerileri oluşturur."""
    # Ürün bazlı analiz
    product_analysis = df.groupby('urun_adi').agg({
        'miktar': ['sum', 'mean', 'std'],
        'tarih': 'count'
    }).round(2)
    
    product_analysis.columns = ['Toplam Satış', 'Ortalama Satış', 'Satış Std', 'Sipariş Sayısı']
    
    # Güvenlik stoku hesaplama (basit yöntem)
    product_analysis['Güvenlik Stoku'] = (product_analysis['Ortalama Satış'] * 1.5).round()
    
    # Yeniden sipariş noktası
    product_analysis['Yeniden Sipariş Noktası'] = (product_analysis['Ortalama Satış'] * 2).round()
    
    # Stok durumu önerisi
    def stock_recommendation(row: pd.Series) -> str:
        if row['Toplam Satış'] > row['Güvenlik Stoku'] * 2:
            return 'Yüksek Stok'
        elif row['Toplam Satış'] < row['Güvenlik Stoku']:
            return 'Stok Yenileme Gerekli'
        else:
            return 'Normal Stok'
    
    product_analysis['Stok Durumu'] = product_analysis.apply(stock_recommendation, axis=1)
    
    return product_analysis

def generate_report(df: pd.DataFrame, forecast_results: Dict[str, Any], stock_analysis: pd.DataFrame) -> str:
    """Kapsamlı interaktif rapor oluşturur."""
    # Gelişmiş rapor şablonu
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>E-Ticaret Satış Analiz Raporu</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0;
                padding: 20px;
                color: #333;
                background-color: #f9f9f9;
            }
            .report-container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            h1 { 
                color: #2c3e50; 
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 2px solid #eee;
                margin-bottom: 30px;
            }
            h2 { 
                color: #3498db; 
                border-left: 4px solid #3498db;
                padding-left: 10px;
                margin-top: 40px;
            }
            h3 {
                color: #555;
                margin-top: 25px;
                border-bottom: 1px solid #eee;
                padding-bottom: 8px;
            }
            .section { 
                margin: 30px 0;
                padding: 20px;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }
            .metric-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: space-between;
                margin-bottom: 30px;
            }
            .metric-card {
                flex: 1;
                min-width: 200px;
                background: #f5f9ff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                text-align: center;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #2980b9;
                margin: 10px 0;
            }
            .metric-label {
                font-size: 14px;
                color: #7f8c8d;
                text-transform: uppercase;
            }
            .metric-change {
                font-size: 14px;
                margin-top: 5px;
            }
            .positive-change {
                color: #27ae60;
            }
            .negative-change {
                color: #e74c3c;
            }
            table { 
                width: 100%; 
                border-collapse: collapse; 
                margin: 20px 0;
                font-size: 14px;
            }
            th, td { 
                border: 1px solid #ddd; 
                padding: 12px; 
                text-align: left; 
            }
            th { 
                background-color: #f2f2f2; 
                position: sticky;
                top: 0;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            tr:hover {
                background-color: #f1f1f1;
            }
            .table-container {
                max-height: 400px;
                overflow-y: auto;
                margin: 20px 0;
            }
            .chart-container {
                width: 100%;
                height: 400px;
                margin: 20px 0;
            }
            .footer {
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #7f8c8d;
                font-size: 12px;
            }
            .recommendation {
                background-color: #e8f4fd;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                border-left: 4px solid #3498db;
            }
            .warning {
                background-color: #fff5e6;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                border-left: 4px solid #e67e22;
            }
            .success {
                background-color: #e9f7ef;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                border-left: 4px solid #27ae60;
            }
            .insight {
                background-color: #eef2f7;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                border-left: 4px solid #9b59b6;
            }
            .grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .grid-item {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                padding: 15px;
            }
            .tabs {
                display: flex;
                border-bottom: 1px solid #ddd;
                margin-bottom: 20px;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                margin-right: 5px;
                border-radius: 5px 5px 0 0;
                background-color: #f2f2f2;
            }
            .tab.active {
                background-color: #3498db;
                color: white;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .progress-container {
                width: 100%;
                height: 20px;
                background-color: #f1f1f1;
                border-radius: 10px;
                margin: 10px 0;
            }
            .progress-bar {
                height: 20px;
                border-radius: 10px;
                background-color: #3498db;
            }
            .product-card {
                display: flex;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                margin-bottom: 15px;
                overflow: hidden;
            }
            .product-info {
                padding: 15px;
                flex: 1;
            }
            .product-title {
                font-weight: bold;
                margin-bottom: 5px;
            }
            .product-stats {
                display: flex;
                gap: 15px;
                color: #666;
                font-size: 14px;
            }
            .product-stat {
                display: flex;
                align-items: center;
            }
            .badge {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                margin-right: 10px;
            }
            .badge-success {
                background-color: #e9f7ef;
                color: #27ae60;
            }
            .badge-warning {
                background-color: #fff5e6;
                color: #e67e22;
            }
            .badge-danger {
                background-color: #fdedeb;
                color: #e74c3c;
            }
            .badge-info {
                background-color: #e8f4fd;
                color: #3498db;
            }
            .data-summary {
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 15px;
                margin-bottom: 20px;
            }
            .data-summary-item {
                flex: 1;
                min-width: 200px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 8px;
                text-align: center;
            }
            .action-list {
                list-style-type: none;
                padding: 0;
            }
            .action-item {
                padding: 12px 15px;
                border-left: 3px solid #3498db;
                background-color: #f8f9fa;
                margin-bottom: 10px;
                border-radius: 0 8px 8px 0;
            }
            .action-item:hover {
                background-color: #eef2f7;
            }
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .kpi-card {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                padding: 20px;
                text-align: center;
            }
            .kpi-title {
                font-size: 16px;
                color: #7f8c8d;
                margin-bottom: 10px;
            }
            .kpi-value {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .kpi-change {
                font-size: 14px;
            }
            
            @media print {
                body { background-color: white; }
                .report-container {
                    box-shadow: none;
                    margin: 0;
                    padding: 0;
                }
                .chart-container { break-inside: avoid; }
                .section { break-inside: avoid; }
            }
        </style>
        <script>
            // Tab navigation için basit script
            document.addEventListener('DOMContentLoaded', function() {
                const tabs = document.querySelectorAll('.tab');
                const tabContents = document.querySelectorAll('.tab-content');
                
                tabs.forEach(tab => {
                    tab.addEventListener('click', () => {
                        const tabId = tab.getAttribute('data-tab');
                        
                        tabs.forEach(t => t.classList.remove('active'));
                        tabContents.forEach(content => content.classList.remove('active'));
                        
                        tab.classList.add('active');
                        document.getElementById(tabId).classList.add('active');
                    });
                });
            });
        </script>
    </head>
    <body>
        <div class="report-container">
            <h1>📊 E-Ticaret Satış Analiz Raporu</h1>
            
            <div class="data-summary">
                <div class="data-summary-item">
                    <p>Rapor Tarihi: {{ generation_date }}</p>
                    <p>Analiz Dönemi: {{ period_description }}</p>
                </div>
                <div class="data-summary-item">
                    <p>Toplam Veri Sayısı: {{ total_records }}</p>
                    <p>Veri Kalitesi: {{ data_quality }}</p>
                </div>
            </div>
            
            <div class="tabs">
                <div class="tab active" data-tab="tab-overview">Genel Bakış</div>
                <div class="tab" data-tab="tab-products">Ürün Analizi</div>
                <div class="tab" data-tab="tab-customers">Müşteri Analizi</div>
                <div class="tab" data-tab="tab-forecasting">Tahminler</div>
                <div class="tab" data-tab="tab-stock">Stok Durumu</div>
                <div class="tab" data-tab="tab-actions">Aksiyon Önerileri</div>
            </div>
            
            <!-- Genel Bakış Tab İçeriği -->
            <div class="tab-content active" id="tab-overview">
                <div class="section">
                    <h2>📈 Performans KPI'ları</h2>
                    <div class="kpi-grid">
                        <div class="kpi-card">
                            <div class="kpi-title">Toplam Satış</div>
                            <div class="kpi-value">{{ total_sales }} TL</div>
                            <div class="kpi-change {{ total_sales_change_class }}">{{ total_sales_change }}</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Müşteri Sayısı</div>
                            <div class="kpi-value">{{ customer_count }}</div>
                            <div class="kpi-change {{ customer_change_class }}">{{ customer_change }}</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Sipariş Sayısı</div>
                            <div class="kpi-value">{{ total_orders }}</div>
                            <div class="kpi-change {{ order_change_class }}">{{ order_change }}</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Sepet Ortalaması</div>
                            <div class="kpi-value">{{ avg_sales }} TL</div>
                            <div class="kpi-change {{ avg_sales_change_class }}">{{ avg_sales_change }}</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Konversiyon Oranı</div>
                            <div class="kpi-value">{{ conversion_rate }}%</div>
                            <div class="kpi-change {{ conversion_change_class }}">{{ conversion_change }}</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Aktif Ürün Sayısı</div>
                            <div class="kpi-value">{{ unique_products }}</div>
                        </div>
                    </div>
                    
                    <div class="insight">
                        <strong>Satış Trendi:</strong> {{ sales_trend_description }}
                    </div>
                    
                    <h3>Özet İstatistikler</h3>
                    <div class="table-container">
                        {{ summary_stats }}
                    </div>
                    
                    <h3>Dönemsel Karşılaştırma</h3>
                    <div class="grid-container">
                        <div class="grid-item">
                            <h4>Günlük Ortalama Satış</h4>
                            <div class="metric-value">{{ daily_avg_sales }} TL</div>
                            <div class="metric-change {{ daily_avg_change_class }}">{{ daily_avg_change }}</div>
                        </div>
                        <div class="grid-item">
                            <h4>Haftalık En Yüksek Satış</h4>
                            <div class="metric-value">{{ weekly_max_sales }} TL</div>
                            <p>{{ best_sales_day }}</p>
                        </div>
                        <div class="grid-item">
                            <h4>En Yoğun Satış Saati</h4>
                            <div class="metric-value">{{ best_sales_hour }}</div>
                            <p>Günün en aktif saati</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Ürün Analizi Tab İçeriği -->
            <div class="tab-content" id="tab-products">
                <div class="section">
                    <h2>🏆 Ürün Performans Analizi</h2>
                    
                    <div class="success">
                        <strong>En Çok Satan Ürün:</strong> {{ top_product }} - Toplam {{ top_product_sales }} adet satış
                    </div>
                    
                    <h3>En Çok Satan 10 Ürün</h3>
                    <div class="table-container">
                        {{ top_products }}
                    </div>
                    
                    <h3>En Yüksek Ciro Ürünleri</h3>
                    <div class="table-container">
                        {{ top_revenue_products }}
                    </div>
                    
                    <h3>Kategori Bazlı Analiz</h3>
                    <div class="table-container">
                        {{ category_performance }}
                    </div>
                    
                    <div class="recommendation">
                        <strong>Ürün Stratejisi:</strong> {{ product_strategy }}
                    </div>
                    
                    <h3>Performansı Düşük Ürünler</h3>
                    <div class="table-container">
                        {{ low_performing_products }}
                    </div>
                </div>
            </div>
            
            <!-- Müşteri Analizi Tab İçeriği -->
            <div class="tab-content" id="tab-customers">
                <div class="section">
                    <h2>👥 Müşteri Segmentasyonu</h2>
                    
                    <h3>Segment Bazlı Analiz</h3>
                    <div class="table-container">
                        {{ segment_data }}
                    </div>
                    
                    <h3>En Değerli 10 Müşteri</h3>
                    <div class="table-container">
                        {{ top_customers }}
                    </div>
                    
                    <h3>Müşteri Davranış Analizi</h3>
                    <div class="grid-container">
                        <div class="grid-item">
                            <h4>Müşteri Yaşam Boyu Değeri (CLV)</h4>
                            <div class="metric-value">{{ avg_customer_value }} TL</div>
                        </div>
                        <div class="grid-item">
                            <h4>Ortalama Sipariş Sıklığı</h4>
                            <div class="metric-value">{{ avg_order_frequency }} gün</div>
                        </div>
                    </div>
                    
                    <div class="recommendation">
                        <strong>Müşteri Stratejisi:</strong> {{ customer_strategy }}
                    </div>
                </div>
            </div>
            
            <!-- Tahminler Tab İçeriği -->
            <div class="tab-content" id="tab-forecasting">
                <div class="section">
                    <h2>🔮 Satış Tahminleri</h2>
                    
                    <div class="metric-container">
                        <div class="metric-card">
                            <div class="metric-label">Prophet Model Tahmini (30 Gün)</div>
                            <div class="metric-value">{{ prophet_forecast }} TL</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">ARIMA Model Tahmini (30 Gün)</div>
                            <div class="metric-value">{{ arima_forecast }} TL</div>
                        </div>
                    </div>
                    
                    <div class="recommendation">
                        <strong>Tahmin Sonuçları:</strong> Gelecek 30 günlük beklenen satış tahmini Prophet modeline göre {{ prophet_forecast }} TL, 
                        ARIMA modeline göre {{ arima_forecast }} TL olarak hesaplanmıştır.
                    </div>
                    
                    <h3>Dönemsel Tahminler</h3>
                    <div class="table-container">
                        {{ forecast_periods }}
                    </div>
                    
                    <h3>Kategori Bazlı Büyüme Tahminleri</h3>
                    <div class="table-container">
                        {{ category_growth_forecast }}
                    </div>
                </div>
            </div>
            
            <!-- Stok Durumu Tab İçeriği -->
            <div class="tab-content" id="tab-stock">
                <div class="section">
                    <h2>📦 Stok Optimizasyonu</h2>
                    
                    <div class="warning">
                        <strong>Stok Uyarısı:</strong> {{ low_stock_count }} ürün için stok yenileme gereklidir.
                    </div>
                    
                    <h3>Stok Durumu Özeti</h3>
                    <div class="grid-container">
                        <div class="grid-item">
                            <h4>Yüksek Stok</h4>
                            <div class="metric-value">{{ high_stock_count }}</div>
                            <p>Ürün</p>
                        </div>
                        <div class="grid-item">
                            <h4>Normal Stok</h4>
                            <div class="metric-value">{{ normal_stock_count }}</div>
                            <p>Ürün</p>
                        </div>
                        <div class="grid-item">
                            <h4>Stok Yenileme Gerekli</h4>
                            <div class="metric-value">{{ low_stock_count }}</div>
                            <p>Ürün</p>
                        </div>
                    </div>
                    
                    <h3>Detaylı Stok Durumu</h3>
                    <div class="table-container">
                        {{ stock_table }}
                    </div>
                    
                    <h3>Acil Sipariş Edilmesi Gereken Ürünler</h3>
                    <div class="table-container">
                        {{ urgent_stock_products }}
                    </div>
                </div>
            </div>
            
            <!-- Aksiyon Önerileri Tab İçeriği -->
            <div class="tab-content" id="tab-actions">
                <div class="section">
                    <h2>📋 Aksiyon Önerileri</h2>
                    
                    <h3>Satış Arttırma Stratejileri</h3>
                    <ul class="action-list">
                        {% for action in sales_actions %}
                        <li class="action-item">{{ action }}</li>
                        {% endfor %}
                    </ul>
                    
                    <h3>Segment Bazlı Müşteri Stratejileri</h3>
                    {% for segment, strategy in segments_strategies.items() %}
                    <div class="recommendation">
                        <strong>{{ segment }}:</strong>
                        <ul>
                            {% for item in strategy.split('\n') %}
                            {% if item.strip() %}
                            <li>{{ item.strip().replace('-', '', 1) }}</li>
                            {% endif %}
                            {% endfor %}
                        </ul>
                    </div>
                    {% endfor %}
                    
                    <h3>Stok Yönetimi Önerileri</h3>
                    <ul class="action-list">
                        {% for action in stock_actions %}
                        <li class="action-item">{{ action }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>Bu rapor {{ generation_date }} tarihinde oluşturulmuştur.</p>
                <p>© E-Ticaret Satış Analizi</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Ek analiz verileri
    current_date = pd.Timestamp.now().strftime("%d.%m.%Y")
    date_range = f"{df['tarih'].min().strftime('%d.%m.%Y')} - {df['tarih'].max().strftime('%d.%m.%Y')}"
    unique_products_count = df['urun_adi'].nunique()
    total_records = len(df)
    
    # Veri kalitesi kontrolü
    missing_data = df.isnull().sum().sum()
    data_quality_score = 100 - (missing_data / (df.size) * 100)
    data_quality = f"%{data_quality_score:.1f} (Eksik Veri: {missing_data})"
    
    # Özet istatistikler
    summary_stats_df = pd.DataFrame({
        'Metrik': ['Toplam Satış', 'Ortalama Satış', 'Minimum Satış', 'Maksimum Satış', 'Standart Sapma', 'Ürün Çeşidi', 'Tarih Aralığı'],
        'Değer': [
            f"₺{df['satis_tutari'].sum():,.2f}",
            f"₺{df['satis_tutari'].mean():,.2f}",
            f"₺{df['satis_tutari'].min():,.2f}",
            f"₺{df['satis_tutari'].max():,.2f}",
            f"₺{df['satis_tutari'].std():,.2f}",
            unique_products_count,
            date_range
        ]
    })
    
    # Stok durumu analizi
    low_stock_items = stock_analysis[stock_analysis['Stok Durumu'] == 'Stok Yenileme Gerekli']
    low_stock_count = len(low_stock_items)
    normal_stock_count = len(stock_analysis[stock_analysis['Stok Durumu'] == 'Normal Stok'])
    high_stock_count = len(stock_analysis[stock_analysis['Stok Durumu'] == 'Yüksek Stok'])
    
    # Acil sipariş ürünleri
    try:
        urgent_stock_products_df = low_stock_items.sort_values('Toplam Satış', ascending=False).head(10)
        urgent_stock_products_html = urgent_stock_products_df.to_html(
            classes='table table-striped',
            float_format=lambda x: '{:,.2f}'.format(x) if isinstance(x, float) else x
        )
    except:
        urgent_stock_products_html = "<p>Acil sipariş edilmesi gereken ürün bulunamadı.</p>"
    
    # Günlük ve haftalık satış analizleri
    try:
        daily_sales = df.groupby(df['tarih'].dt.date)['satis_tutari'].sum()
        daily_avg_sales = daily_sales.mean()
        weekly_max_sales = df.groupby(df['tarih'].dt.isocalendar().week)['satis_tutari'].sum().max()
        
        # En iyi satış günü ve saati
        best_day_idx = daily_sales.idxmax()
        best_day_name = pd.Timestamp(best_day_idx).strftime('%A')
        best_day_formatted = pd.Timestamp(best_day_idx).strftime('%d.%m.%Y')
        best_sales_day = f"{best_day_name}, {best_day_formatted}"
        
        # En iyi satış saati
        if 'satis_saati' not in df.columns and 'tarih' in df.columns:
            df['satis_saati'] = df['tarih'].dt.hour
            
        best_sales_hour = f"{df.groupby('satis_saati')['satis_tutari'].sum().idxmax()}:00"
    except:
        daily_avg_sales = 0
        weekly_max_sales = 0
        best_sales_day = "Veri yok"
        best_sales_hour = "Veri yok"
    
    # En çok satan ürünler
    top_products_data = None
    top_product_name = ""
    top_product_sales = 0
    
    try:
        # En çok satan 10 ürün
        top_products = df.groupby('urun_adi')['miktar'].sum().sort_values(ascending=False).head(10)
        top_product_name = top_products.index[0]
        top_product_sales = top_products.iloc[0]
        
        # Tablo için dataframe oluştur
        top_products_df = df.groupby('urun_adi').agg({
            'miktar': 'sum',
            'satis_tutari': ['sum', 'mean']
        }).round(2)
        
        top_products_df.columns = ['Toplam Satış Miktarı', 'Toplam Satış Tutarı', 'Ortalama Satış Tutarı']
        top_products_df = top_products_df.sort_values('Toplam Satış Miktarı', ascending=False).head(10)
        
        # En yüksek cirolu ürünler
        top_revenue_df = top_products_df.sort_values('Toplam Satış Tutarı', ascending=False).head(10)
        
        # Düşük performanslı ürünler (belirli bir eşiğin altında satış yapan ürünler)
        sales_threshold = top_products_df['Toplam Satış Miktarı'].median() * 0.3
        low_performing_df = top_products_df[top_products_df['Toplam Satış Miktarı'] < sales_threshold].sort_values('Toplam Satış Miktarı')
        
        # HTML tabloları oluştur
        top_products_data = top_products_df.to_html(
            classes='table table-striped',
            float_format=lambda x: '{:,.2f}'.format(x) if isinstance(x, float) else x
        )
        
        top_revenue_products = top_revenue_df.to_html(
            classes='table table-striped',
            float_format=lambda x: '{:,.2f}'.format(x) if isinstance(x, float) else x
        )
        
        low_performing_products = low_performing_df.to_html(
            classes='table table-striped',
            float_format=lambda x: '{:,.2f}'.format(x) if isinstance(x, float) else x
        ) if not low_performing_df.empty else "<p>Düşük performanslı ürün tespit edilmedi.</p>"
    except:
        top_products_data = "<p>Ürün analizi yapılırken bir hata oluştu.</p>"
        top_revenue_products = "<p>Ciro analizi yapılırken bir hata oluştu.</p>"
        low_performing_products = "<p>Performans analizi yapılırken bir hata oluştu.</p>"
    
    # Kategori performansı
    category_performance = ""
    if 'kategori' in df.columns:
        try:
            category_df = df.groupby('kategori').agg({
                'satis_tutari': ['sum', 'mean', 'count'],
                'miktar': 'sum'
            }).round(2)
            
            category_df.columns = ['Toplam Satış', 'Ortalama Satış', 'Sipariş Sayısı', 'Toplam Miktar']
            category_performance = category_df.sort_values('Toplam Satış', ascending=False).to_html(
                classes='table table-striped',
                float_format=lambda x: '{:,.2f}'.format(x) if isinstance(x, float) else x
            )
        except:
            category_performance = "<p>Kategori analizi yapılırken bir hata oluştu.</p>"
    else:
        category_performance = "<p>Kategori verisi bulunamadı.</p>"
    
    # Müşteri segmentasyonu
    segment_data = None
    top_customers = None
    vip_percentage = 0
    vip_sales_percentage = 0
    avg_customer_value = 0
    avg_order_frequency = 0
    new_customer_rate = 0
    customer_count = 0
    
    if 'musteri_id' in df.columns:
        try:
            # Toplam müşteri sayısı
            customer_count = df['musteri_id'].nunique()
            
            # RFM analizi
            rfm_data = calculate_rfm(df)
            
            # Segment metrikleri
            segment_metrics = rfm_data.groupby('Segment').agg({
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': ['mean', 'sum'] # Hem ortalama hem toplam ekleyelim
            })
            
            # MultiIndex'i düzleştirelim
            segment_metrics.columns = ['Ort. Recency (gün)', 'Ort. Frequency (sipariş)', 
                                      'Ort. Monetary (₺)', 'Toplam Satış (₺)']
            
            segment_metrics = segment_metrics.round(2)
            
            # VIP müşteri istatistikleri
            vip_customers = rfm_data[rfm_data['Segment'] == 'VIP Müşteriler']
            total_customers = len(rfm_data)
            total_sales = rfm_data['monetary'].sum()
            
            vip_percentage = round((len(vip_customers) / total_customers) * 100, 1) if total_customers > 0 else 0
            vip_sales_percentage = round((vip_customers['monetary'].sum() / total_sales) * 100, 1) if total_sales > 0 else 0
            
            # En değerli müşteriler
            top_customers_df = rfm_data.sort_values('monetary', ascending=False).head(10)
            
            # Müşteri yaşam boyu değeri ve sipariş sıklığı
            avg_customer_value = rfm_data['monetary'].mean()
            avg_order_frequency = rfm_data['frequency'].mean()
            
            # HTML tabloları oluştur
            segment_data = segment_metrics.to_html(
                classes='table table-striped',
                float_format=lambda x: '{:,.2f}'.format(x) if isinstance(x, float) else x
            )
            
            top_customers = top_customers_df.to_html(
                classes='table table-striped',
                float_format=lambda x: '{:,.2f}'.format(x) if isinstance(x, float) else x
            )
        except Exception as e:
            segment_data = f"<p>Müşteri segmentasyonu yapılırken bir hata oluştu: {str(e)}</p>"
            top_customers = "<p>Müşteri analizi yapılırken bir hata oluştu.</p>"
            vip_percentage = 0
            vip_sales_percentage = 0
            
    else:
        segment_data = "<p>Müşteri analizi için 'musteri_id' sütunu gereklidir.</p>"
        top_customers = "<p>Müşteri analizi için 'musteri_id' sütunu gereklidir.</p>"
        customer_strategy = "Müşteri analizi için veri formatınızı kontrol ediniz. Müşteri bazlı analiz yapabilmek için 'musteri_id' sütunu gereklidir."
    
    # Satış trendi ve değişim oranları
    total_sales_change = "+%0.0"
    total_sales_change_class = ""
    customer_change = "+%0.0"
    customer_change_class = ""
    order_change = "+%0.0"
    order_change_class = ""
    avg_sales_change = "+%0.0"
    avg_sales_change_class = ""
    conversion_change = "+%0.0"
    conversion_change_class = ""
    daily_avg_change = "+%0.0"
    daily_avg_change_class = ""
    
    # Varsayılan değerler
    conversion_rate = 0
    sales_trend_description = "Satış trendi analizi için yeterli veri bulunamadı."
    
    # Dönemsel tahminler
    forecast_periods_html = "<p>Dönemsel tahmin analizi yapılırken bir hata oluştu.</p>"
    category_growth_forecast_html = "<p>Kategori büyüme tahmini yapılırken bir hata oluştu.</p>"
    
    try:
        # Son 30 günlük tahminleri al
        forecast_daily = forecast_results['prophet_forecast'].tail(30)
        
        # Haftalık, aylık ve çeyreklik tahminler
        weekly_forecast = forecast_daily.groupby(pd.Grouper(key='ds', freq='W'))['yhat'].sum().reset_index()
        weekly_forecast.columns = ['Hafta', 'Tahmini Satış (TL)']
        
        monthly_forecast = forecast_daily.groupby(pd.Grouper(key='ds', freq='M'))['yhat'].sum().reset_index()
        monthly_forecast.columns = ['Ay', 'Tahmini Satış (TL)']
        
        # Tahmin tabloları
        forecast_periods_df = pd.DataFrame({
            'Dönem': ['Haftalık (Sonraki 4 Hafta)', 'Yıllık (Sonraki Yıl)'],
            'Tahmini Satış (TL)': [
                f"₺{weekly_forecast['Tahmini Satış (TL)'].sum():,.2f}",
                f"₺{monthly_forecast['Tahmini Satış (TL)'].sum() * 12:,.2f}"
            ]
        })
        
        forecast_periods_html = forecast_periods_df.to_html(
            classes='table table-striped',
            index=False
        )
        
        # Kategori büyüme tahminleri
        if 'kategori' in df.columns:
            try:
                # Son 3 aylık kategori bazlı büyüme oranlarını hesapla
                df['ay'] = df['tarih'].dt.to_period('M')
                last_3months = df.groupby(['kategori', 'ay'])['satis_tutari'].sum().unstack().fillna(0)
                
                if len(last_3months.columns) >= 2:
                    growth_rates = ((last_3months[last_3months.columns[-1]] / last_3months[last_3months.columns[-2]]) - 1) * 100
                    
                    category_growth_df = pd.DataFrame({
                        'Kategori': growth_rates.index,
                        'Son Ay Büyüme (%)': growth_rates.values,
                        'Tahmini Sonraki Ay (%)': growth_rates.values * 0.8  # Basit bir tahmin
                    }).sort_values('Tahmini Sonraki Ay (%)', ascending=False)
                    
                    category_growth_forecast_html = category_growth_df.to_html(
                        classes='table table-striped',
                        float_format=lambda x: '{:+.2f}%'.format(x) if isinstance(x, float) else x,
                        index=False
                    )
            except:
                category_growth_forecast_html = "<p>Kategori büyüme tahmini için yeterli veri bulunamadı.</p>"
    except:
        forecast_periods_html = "<p>Dönemsel tahmin analizi için yeterli veri bulunamadı.</p>"
    
    # Son 30 günlük satış trendi
    try:
        df['tarih_gun'] = df['tarih'].dt.date
        last_30days = df.groupby('tarih_gun')['satis_tutari'].sum().tail(30)
        
        if len(last_30days) > 15:
            # Basit doğrusal regresyon ile trend analizi
            x = np.arange(len(last_30days))
            y = last_30days.values
            slope, _, _, _, _ = stats.linregress(x, y)
            
            if slope > 0:
                sales_trend_description = f"Son 30 günde satışlarda %{slope*100/y.mean():.1f} artış trendi görülmektedir. Bu artış devam ederse, gelecek ayda satışların daha da yükselmesi beklenebilir."
            elif slope < 0:
                sales_trend_description = f"Son 30 günde satışlarda %{-slope*100/y.mean():.1f} düşüş trendi görülmektedir. Satışları artırmak için pazarlama stratejileri gözden geçirilmelidir."
            else:
                sales_trend_description = "Son 30 günde satışlar stabil seyretmektedir. Büyüme için yeni stratejiler geliştirilebilir."
    except:
        pass
    
    # Aksiyon önerileri
    sales_actions = [
        "Satışları artırmak için en çok satan ürünlerde kampanya düzenleyin.",
        f"'{top_product_name}' ürününün fiyatını optimize ederek satışlarını daha da artırabilirsiniz.",
        "Düşük performanslı ürünleri indirime alarak stoklarını eritin.",
        f"En yoğun satış saati olan {best_sales_hour}'da özel kampanyalar düzenleyin.",
        "VIP müşterilerinize özel indirim ve kampanyalar sunarak onları elde tutun.",
        "Yeni müşteri kazanımı için 'arkadaşını getir' kampanyası düzenleyin."
    ]
    
    stock_actions = [
        f"{low_stock_count} ürün için acil stok yenileme yapın.",
        "Yüksek stoktaki ürünler için özel kampanyalar düzenleyin.",
        "Stok devir hızını artırmak için fiyatlandırma stratejilerini optimize edin.",
        "Mevsimsel ürünlerin stoklarını önceden planlayın.",
        "Tedarik zincirini gözden geçirerek stok maliyetlerini düşürün."
    ]
    
    # Ürün ve müşteri stratejileri
    product_strategy = f"En çok satan '{top_product_name}' ürününü öne çıkarın ve buna benzer ürünleri stoklara ekleyin. Düşük performanslı ürünleri indirime alarak stok devir hızını artırın ve kategori bazlı performans analizine göre ürün çeşitliliğini optimize edin."
    
    customer_strategy = f"VIP müşterileriniz toplam satışların {vip_sales_percentage:.1f}%'sini oluşturuyor. Bu müşterilere özel avantajlar sunarak sadakatlerini artırın. Risk altındaki müşterileri geri kazanmak için özel kampanyalar düzenleyin ve yeni müşteri kazanımı için referans programları oluşturun."
    
    # Pazarlama stratejileri
    segments_strategies = {
        'VIP Müşteriler': """
        - Özel indirimler ve kampanyalar sunun
        - Sadakat programları ve premium üyelik teklifleri
        - Kişiselleştirilmiş ürün önerileri
        - Yeni ürünleri ilk deneme fırsatı
        - Özel müşteri hizmetleri ve ayrıcalıklar
        - Doğum günü veya yıldönümlerinde hediyeler
        """,
        
        'Sadık Müşteriler': """
        - Düzenli indirim kuponları
        - Özel müşteri etkinlikleri
        - Referans programları
        - Kişiselleştirilmiş e-posta pazarlaması
        - Puan biriktirme ve ödül sistemi
        - Sipariş sonrası takip ve memnuniyet anketleri
        """,
        
        'Potansiyel Müşteriler': """
        - Sınırlı süreli kampanyalar
        - Ürün çeşitliliği tanıtımları
        - 'Bunları da beğenebilirsiniz' önerileri
        - İndirimli ürün demetleri
        - İlk siparişe özel indirimler
        - Eğitici içerikler ve ürün kullanım tavsiyeleri
        """,
        
        'Risk Altındaki Müşteriler': """
        - Geri kazanım kampanyaları
        - Büyük indirimler
        - Memnuniyet anketleri
        - 'Sizi özledik' mesajları
        - Son şans indirimleri
        - Ürün iade/değişim kolaylığı
        """
    }
    
    # Şablonu doldur
    html_content = Template(template).render(
        # Genel metrikler
        total_sales=f"{df['satis_tutari'].sum():,.2f}",
        avg_sales=f"{df['satis_tutari'].mean():,.2f}",
        total_orders=f"{len(df):,}",
        unique_products=f"{unique_products_count:,}",
        period_description=date_range,
        
        # Özet istatistikler
        total_records=f"{total_records:,}",
        data_quality=data_quality,
        summary_stats=summary_stats_df.to_html(classes='table table-striped', index=False),
        
        # Dönemsel analizler
        daily_avg_sales=f"{daily_avg_sales:,.2f}",
        weekly_max_sales=f"{weekly_max_sales:,.2f}",
        best_sales_day=best_sales_day,
        best_sales_hour=best_sales_hour,
        
        # Değişim oranları
        total_sales_change=total_sales_change,
        total_sales_change_class=total_sales_change_class,
        customer_change=customer_change,
        customer_change_class=customer_change_class,
        order_change=order_change,
        order_change_class=order_change_class,
        avg_sales_change=avg_sales_change,
        avg_sales_change_class=avg_sales_change_class,
        conversion_rate=conversion_rate,
        conversion_change=conversion_change,
        conversion_change_class=conversion_change_class,
        daily_avg_change=daily_avg_change,
        daily_avg_change_class=daily_avg_change_class,
        
        # Müşteri analizi
        customer_count=f"{customer_count:,}",
        vip_percentage="",
        vip_sales_percentage="",
        segment_data=segment_data,
        top_customers=top_customers,
        avg_customer_value=f"{avg_customer_value:,.2f}",
        avg_order_frequency=f"{avg_order_frequency:.1f}",
        new_customer_rate="",
        
        # Ürün analizi
        top_products=top_products_data,
        top_product=top_product_name,
        top_product_sales=f"{top_product_sales:,}",
        top_revenue_products=top_revenue_products,
        low_performing_products=low_performing_products,
        category_performance=category_performance,
        product_strategy=product_strategy,
        
        # Tahmin analizi
        prophet_forecast=f"{forecast_results['prophet_forecast']['yhat'].iloc[-30:].mean():,.2f}",
        arima_forecast=f"{forecast_results['arima_forecast'].mean():,.2f}",
        forecast_periods=forecast_periods_html,
        category_growth_forecast=category_growth_forecast_html,
        sales_trend_description=sales_trend_description,
        
        # Stok analizi
        stock_table=stock_analysis.to_html(classes='table table-striped'),
        low_stock_count=low_stock_count,
        normal_stock_count=normal_stock_count,
        high_stock_count=high_stock_count,
        urgent_stock_products=urgent_stock_products_html,
        
        # Stratejiler ve öneriler
        customer_strategy=customer_strategy,
        sales_actions=sales_actions,
        stock_actions=stock_actions,
        segments_strategies=segments_strategies,
        
        # Diğer
        generation_date=current_date
    )
    
    return html_content

def comparative_analysis(df):
    st.header("🔄 Karşılaştırmalı Analiz")
    st.info("""
    Bu bölüm, farklı dönemlerdeki satış performansınızı karşılaştırmanızı sağlar. 
    Örneğin, bu ayın satışlarını geçen ay ile karşılaştırabilir veya bu yılın performansını geçen yıl ile ölçebilirsiniz.
    """)
    
    # Dönem seçimi
    col1, col2 = st.columns(2)
    with col1:
        current_period = st.selectbox(
            "Karşılaştırılacak Dönem",
            options=['Bu Ay', 'Bu Hafta', 'Bu Yıl', 'Son 30 Gün', 'Son 90 Gün'],
            key='current_period'
        )
    with col2:
        compare_with = st.selectbox(
            "Karşılaştırılacak Önceki Dönem",
            options=['Geçen Ay', 'Geçen Hafta', 'Geçen Yıl', 'Önceki 30 Gün', 'Önceki 90 Gün'],
            key='compare_with'
        )
    
    # Dönemleri hesapla
    def get_date_range(period):
        end_date = df['tarih'].max()
        if period == 'Bu Ay':
            start_date = end_date.replace(day=1)
        elif period == 'Bu Hafta':
            start_date = end_date - pd.Timedelta(days=end_date.weekday())
        elif period == 'Bu Yıl':
            start_date = end_date.replace(month=1, day=1)
        elif period == 'Son 30 Gün':
            start_date = end_date - pd.Timedelta(days=30)
        elif period == 'Son 90 Gün':
            start_date = end_date - pd.Timedelta(days=90)
        elif period == 'Geçen Ay':
            start_date = (end_date.replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
            end_date = end_date.replace(day=1) - pd.Timedelta(days=1)
        elif period == 'Geçen Hafta':
            start_date = end_date - pd.Timedelta(days=end_date.weekday() + 7)
            end_date = end_date - pd.Timedelta(days=end_date.weekday() + 1)
        elif period == 'Geçen Yıl':
            start_date = end_date.replace(year=end_date.year-1, month=1, day=1)
            end_date = end_date.replace(year=end_date.year-1, month=12, day=31)
        elif period == 'Önceki 30 Gün':
            start_date = end_date - pd.Timedelta(days=60)
            end_date = end_date - pd.Timedelta(days=30)
        elif period == 'Önceki 90 Gün':
            start_date = end_date - pd.Timedelta(days=180)
            end_date = end_date - pd.Timedelta(days=90)
        return start_date, end_date
    
    current_start, current_end = get_date_range(current_period)
    compare_start, compare_end = get_date_range(compare_with)
    
    # Dönem verilerini filtrele
    current_data = df[(df['tarih'] >= current_start) & (df['tarih'] <= current_end)]
    compare_data = df[(df['tarih'] >= compare_start) & (df['tarih'] <= compare_end)]
    
    # Karşılaştırma metrikleri
    metrics = {
        'Ortalama Satış': ('satis_tutari', 'mean'),
        'Sipariş Sayısı': ('siparis_id', 'count') if 'siparis_id' in df.columns else ('tarih', 'count'),
        'Toplam Satış': ('satis_tutari', 'sum'),
        'Ortalama Sepet': ('satis_tutari', 'mean')
    }
    
    # Metrikleri hesapla ve göster
    st.subheader("Dönemsel Karşılaştırma Metrikleri")
    col1, col2, col3, col4 = st.columns(4)  # 4 sütun oluştur
    
    for i, (metric_name, (column, operation)) in enumerate(metrics.items()):
        current_value = getattr(current_data[column], operation)()
        compare_value = getattr(compare_data[column], operation)()
        change = ((current_value - compare_value) / compare_value * 100) if compare_value != 0 else 0
        
        with [col1, col2, col3, col4][i]:  # Her metrik için ayrı sütun
            st.metric(
                metric_name,
                f"₺{current_value:,.2f}" if 'satis' in column else f"{current_value:,.0f}",
                f"%{change:,.1f}"
            )
    
    # Karşılaştırmalı grafikler
    st.subheader("Dönemsel Karşılaştırma Grafikleri")
    
    # Günlük satış karşılaştırması
    daily_current = current_data.groupby('tarih')['satis_tutari'].sum()
    daily_compare = compare_data.groupby('tarih')['satis_tutari'].sum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_current.index, 
        y=daily_current.values,
        name=current_period, 
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=daily_compare.index, 
        y=daily_compare.values,
        name=compare_with, 
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title='Günlük Satış Karşılaştırması',
        xaxis_title='Tarih',
        yaxis_title='Satış Tutarı (₺)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Kategori bazlı karşılaştırma (eğer kategori sütunu varsa)
    if 'kategori' in df.columns:
        st.subheader("Kategori Bazlı Karşılaştırma")
        
        category_current = current_data.groupby('kategori')['satis_tutari'].sum()
        category_compare = compare_data.groupby('kategori')['satis_tutari'].sum()
        
        # Kategori büyüme oranları
        category_growth = pd.DataFrame({
            'Mevcut Dönem': category_current,
            'Önceki Dönem': category_compare
        }).fillna(0)
        
        category_growth['Büyüme Oranı'] = ((category_growth['Mevcut Dönem'] - category_growth['Önceki Dönem']) / 
                                          category_growth['Önceki Dönem'] * 100).fillna(0)
        
        # Kategori karşılaştırma grafiği
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=category_growth.index,
            y=category_growth['Mevcut Dönem'],
            name=current_period,
            marker_color='blue'
        ))
        fig.add_trace(go.Bar(
            x=category_growth.index,
            y=category_growth['Önceki Dönem'],
            name=compare_with,
            marker_color='gray'
        ))
        
        fig.update_layout(
            title='Kategori Bazlı Satış Karşılaştırması',
            xaxis_title='Kategori',
            yaxis_title='Satış Tutarı (₺)',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Kategori büyüme oranları tablosu
        st.subheader("Kategori Büyüme Oranları")
        category_growth = category_growth.sort_values('Büyüme Oranı', ascending=False)
        st.dataframe(
            category_growth.style.format({
                'Mevcut Dönem': '₺{:,.2f}',
                'Önceki Dönem': '₺{:,.2f}',
                'Büyüme Oranı': '{:,.1f}%'
            })
        )

def analyze_products(df):
    """Ürün bazlı analiz yapar."""
    st.header("🏆 En Çok Satan Ürünler")
    st.info("""
    En çok satan ürünlerinizi görün ve hangi ürünlerin daha iyi performans gösterdiğini analiz edin.
    Bu bilgi, stok yönetimi ve ürün stratejilerinizi belirlemenize yardımcı olur.
    """)
    
    if len(df) == 0:
        st.warning("Bu filtre için veri bulunamadı.")
        return
    
    # En çok satan ürünler
    try:
        en_cok_satan = df.groupby('urun_adi')['miktar'].sum().sort_values(ascending=False).head(10)
        
        # Bar grafiği
        fig = px.bar(
            en_cok_satan,
            x=en_cok_satan.index,
            y=en_cok_satan.values,
            title='En Çok Satan 10 Ürün',
            labels={'x': 'Ürün Adı', 'y': 'Satış Miktarı'},
            color=en_cok_satan.values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Ürün detayları
        st.subheader("Ürün Detayları")
        product_details = df.groupby('urun_adi').agg({
            'miktar': 'sum',
            'satis_tutari': ['sum', 'mean'],
            'tarih': 'count'
        }).round(2)
        
        product_details.columns = ['Toplam Satış Miktarı', 'Toplam Satış Tutarı', 'Ortalama Satış Tutarı', 'Sipariş Sayısı']
        product_details = product_details.sort_values('Toplam Satış Tutarı', ascending=False)
        
        st.dataframe(
            product_details.style.format({
                'Toplam Satış Tutarı': '₺{:,.2f}',
                'Ortalama Satış Tutarı': '₺{:,.2f}'
            }).background_gradient(subset=['Toplam Satış Miktarı'], cmap='Blues')
        )
        
        # En karlı ürünler analizi
        st.subheader("En Yüksek Cirolu Ürünler")
        en_karli = product_details.sort_values('Toplam Satış Tutarı', ascending=False).head(10)
        
        fig = px.bar(
            en_karli,
            x=en_karli.index,
            y='Toplam Satış Tutarı',
            title='En Yüksek Cirolu 10 Ürün',
            labels={'x': 'Ürün Adı', 'y': 'Satış Tutarı (₺)'},
            color='Toplam Satış Tutarı',
            color_continuous_scale='Greens'
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Ortalama sepet tutarı yüksek olan ürünler
        st.subheader("Yüksek Ortalama Satış Tutarı Olan Ürünler")
        yuksek_ortalama = product_details[product_details['Sipariş Sayısı'] >= 3].sort_values('Ortalama Satış Tutarı', ascending=False).head(10)
        
        if not yuksek_ortalama.empty:
            fig = px.bar(
                yuksek_ortalama,
                x=yuksek_ortalama.index,
                y='Ortalama Satış Tutarı',
                title='Yüksek Ortalama Satış Tutarı Olan Ürünler',
                labels={'x': 'Ürün Adı', 'y': 'Ortalama Satış Tutarı (₺)'},
                color='Ortalama Satış Tutarı',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Yeterli sipariş sayısı olan ürün bulunamadı.")
        
    except Exception as e:
        st.error(f"Ürün analizi sırasında bir hata oluştu: {str(e)}")
        st.info("Veri formatınızı kontrol edin ve yeniden deneyin.")

# Ana uygulama
st.title("📊 E-Ticaret Satış Analizi")
st.markdown("""
Bu uygulama, e-ticaret satış verilerinizi analiz etmenize ve değerli içgörüler elde etmenize yardımcı olur.
Aşağıdaki özellikleri kullanarak satış performansınızı detaylı olarak inceleyebilirsiniz:
""")

# Veri yükleme
uploaded_file = st.file_uploader(
    "Veri dosyasını yükleyin",
    type=['csv', 'xlsx', 'xls', 'json'],
    help="CSV, Excel veya JSON formatında veri dosyası yükleyin. Dosyanızda 'tarih', 'urun_adi', 'miktar' ve 'satis_tutari' sütunları bulunmalıdır."
)

if uploaded_file is not None:
    st.info("""
    Veri dosyanız başarıyla yüklendi! 
    Sol menüden filtreleme seçeneklerini kullanarak analizlerinizi özelleştirebilirsiniz.
    """)
    
    df = load_data(uploaded_file)
    
    if df is not None:
        # Veri doğrulama
        is_valid, message = validate_dataframe(df)
        if not is_valid:
            st.error(message)
            st.stop()
        else:
            st.success(message)
        
        # Tarih dönüşümü
        df = detect_and_convert_date(df)
        
        # Veri önizleme
        with st.expander("Veri Önizleme", expanded=True):
            st.dataframe(df.head())
            st.write(f"Toplam Satır Sayısı: {len(df)}")
            st.write("Sütun Bilgileri:")
            st.write(df.dtypes)
        
        # Filtreleme seçenekleri
        st.sidebar.header("🔍 Filtreleme Seçenekleri")
        
        # Tarih aralığı
        min_date = df['tarih'].min()
        max_date = df['tarih'].max()
        date_range = st.sidebar.date_input(
            "Tarih Aralığı",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Kategori filtresi
        if 'kategori' in df.columns:
            categories = ['Tümü'] + sorted(df['kategori'].unique().tolist())
            selected_category = st.sidebar.selectbox("Kategori", categories)
        
        # Müşteri segmenti filtresi
        if 'musteri_id' in df.columns:
            rfm = calculate_rfm(df)
            segments = ['Tümü'] + sorted(rfm['Segment'].unique().tolist())
            selected_segment = st.sidebar.selectbox("Müşteri Segmenti", segments)
        
        # Filtreleri uygula
        mask = (df['tarih'].dt.date >= date_range[0]) & (df['tarih'].dt.date <= date_range[1])
        filtered_df = df.loc[mask]
        
        if 'kategori' in df.columns and selected_category != 'Tümü':
            filtered_df = filtered_df[filtered_df['kategori'] == selected_category]
        
        if 'musteri_id' in df.columns and selected_segment != 'Tümü':
            segment_customers = rfm[rfm['Segment'] == selected_segment].index
            filtered_df = filtered_df[filtered_df['musteri_id'].isin(segment_customers)]
        
        # Ana metrikler
        col1, col2, col3 = st.columns(3)
        
        with col1:
            toplam_satis = filtered_df['satis_tutari'].sum()
            st.metric("Toplam Satış", f"₺{toplam_satis:,.2f}")
        
        with col2:
            ortalama_satis = filtered_df['satis_tutari'].mean()
            st.metric("Ortalama Satış", f"₺{ortalama_satis:,.2f}")
        
        with col3:
            toplam_siparis = len(filtered_df)
            st.metric("Toplam Sipariş", f"{toplam_siparis:,}")
        
        # Karşılaştırmalı Analiz
        if 'tarih' in df.columns:
            st.markdown("---")  # Ayırıcı çizgi
            comparative_analysis(filtered_df)
            st.markdown("---")  # Ayırıcı çizgi
        
        # Zaman Serisi Analizi
        st.header("📈 Zaman Serisi Analizi")
        st.info("""
        Bu analiz, satışlarınızın zaman içindeki değişimini gösterir. 
        Günlük, haftalık ve aylık trendleri görerek satışlarınızdaki artış/azalışları takip edebilirsiniz.
        """)
        analyze_time_series(filtered_df)
        
        # Satış yoğunluğu ısı haritası
        st.subheader("Satış Yoğunluğu")
        heatmap_fig = create_sales_heatmap(filtered_df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Kategori Analizi
        if 'kategori' in df.columns:
            st.header("📊 Kategori Analizi")
            st.info("""
            Kategori bazlı satış performansınızı inceleyin. 
            Hangi kategorilerin daha iyi performans gösterdiğini ve hangi kategorilerde iyileştirme yapabileceğinizi görün.
            """)
            
            # Analiz verilerini al
            category_data = analyze_categories(filtered_df)
            
            if category_data:
                # Kategori metrikleri
                st.subheader("Kategori Bazlı Performans")
                st.dataframe(
                    category_data['metrics'].style.format({
                        'Toplam Satış': '₺{:,.2f}',
                        'Ortalama Satış': '₺{:,.2f}',
                        'Toplam Miktar': '{:,.0f}',
                        'Sipariş Sayısı': '{:,.0f}'
                    })
                )
                
                # Kategori performans grafiği
                st.subheader("Kategori Satış Dağılımı")
                fig = px.pie(
                    values=category_data['metrics']['Toplam Satış'],
                    names=category_data['metrics'].index,
                    title='Kategori Bazlı Satış Dağılımı'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Kategori büyüme grafiği
                st.subheader("Kategori Büyüme Analizi")
                
                # Pivot tablosu oluştur
                category_pivot = category_data['growth'].pivot(
                    index='tarih', 
                    columns='kategori', 
                    values='satis_tutari'
                )
                
                # Çizgi grafiği
                fig = px.line(
                    category_pivot,
                    title='Kategori Bazlı Satış Trendi',
                    labels={'value': 'Satış Tutarı (₺)', 'tarih': 'Tarih'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Büyüme oranları tablosu
                st.subheader("Kategori Aylık Büyüme Oranları")
                growth_pivot = category_data['growth'].pivot(
                    index='tarih', 
                    columns='kategori', 
                    values='growth'
                ).fillna(0)
                
                st.dataframe(
                    growth_pivot.style.format('{:,.1f}%').background_gradient(
                        cmap='RdYlGn', axis=None
                    )
                )
        
        # RFM Analizi
        if 'musteri_id' in df.columns:
            st.header("👥 Müşteri Segmentasyonu (RFM Analizi)")
            st.info("""
            RFM analizi, müşterilerinizi değerlerine göre segmentlere ayırır:
            - Recency (Yenilik): Son alışverişten bu yana geçen süre
            - Frequency (Sıklık): Alışveriş sıklığı
            - Monetary (Parasal): Toplam harcama tutarı
            
            Bu bilgilerle müşterilerinize özel pazarlama stratejileri geliştirebilirsiniz.
            """)
            
            try:
                # RFM analizi yap
                rfm_data = calculate_rfm(filtered_df)
                
                # RFM metrikleri
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Ortalama Recency", f"{rfm_data['recency'].mean():.1f} gün")
                
                with col2:
                    st.metric("Ortalama Frequency", f"{rfm_data['frequency'].mean():.1f} sipariş")
                
                with col3:
                    st.metric("Ortalama Monetary", f"₺{rfm_data['monetary'].mean():,.2f}")
                
                # Segment dağılımı
                st.subheader("Müşteri Segmentleri Dağılımı")
                segment_counts = rfm_data['Segment'].value_counts()
                
                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title='Müşteri Segmentleri Dağılımı',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Segment detayları
                st.subheader("Segment Detayları")
                segment_metrics = rfm_data.groupby('Segment').agg({
                    'recency': 'mean',
                    'frequency': 'mean',
                    'monetary': 'mean'
                }).round(2)
                
                segment_metrics.columns = ['Ortalama Recency (gün)', 'Ortalama Frequency (sipariş)', 'Ortalama Monetary (₺)']
                segment_metrics = segment_metrics.sort_values('Ortalama Monetary (₺)', ascending=False)
                
                st.dataframe(
                    segment_metrics.style.format({
                        'Ortalama Monetary (₺)': '₺{:,.2f}',
                        'Ortalama Recency (gün)': '{:.1f}',
                        'Ortalama Frequency (sipariş)': '{:.1f}'
                    })
                )
                
                # RFM Skorları ve müşteriler
                st.subheader("RFM Skorları")
                
                # Kullanıcıya segment seçtir
                selected_segment_for_detail = st.selectbox(
                    "Detaylı müşteri listesi için segment seçin:",
                    options=sorted(rfm_data['Segment'].unique())
                )
                
                # Seçilen segmente ait müşterileri göster
                segment_customers = rfm_data[rfm_data['Segment'] == selected_segment_for_detail]
                segment_customers = segment_customers.sort_values('monetary', ascending=False)
                
                st.dataframe(
                    segment_customers.style.format({
                        'recency': '{:.0f} gün',
                        'frequency': '{:.0f}',
                        'monetary': '₺{:,.2f}'
                    })
                )
                
                # Pazarlama önerileri
                st.subheader("Pazarlama Stratejisi Önerileri")
                
                segments_strategies = {
                    'VIP Müşteriler': """
                    - Özel indirimler ve kampanyalar sunun
                    - Sadakat programları ve premium üyelik teklifleri
                    - Kişiselleştirilmiş ürün önerileri
                    - Yeni ürünleri ilk deneme fırsatı
                    - Özel müşteri hizmetleri ve ayrıcalıklar
                    - Doğum günü veya yıldönümlerinde hediyeler
                    """,
                    
                    'Sadık Müşteriler': """
                    - Düzenli indirim kuponları
                    - Özel müşteri etkinlikleri
                    - Referans programları
                    - Kişiselleştirilmiş e-posta pazarlaması
                    - Puan biriktirme ve ödül sistemi
                    - Sipariş sonrası takip ve memnuniyet anketleri
                    """,
                    
                    'Potansiyel Müşteriler': """
                    - Sınırlı süreli kampanyalar
                    - Ürün çeşitliliği tanıtımları
                    - 'Bunları da beğenebilirsiniz' önerileri
                    - İndirimli ürün demetleri
                    - İlk siparişe özel indirimler
                    - Eğitici içerikler ve ürün kullanım tavsiyeleri
                    """,
                    
                    'Risk Altındaki Müşteriler': """
                    - Geri kazanım kampanyaları
                    - Büyük indirimler
                    - Memnuniyet anketleri
                    - 'Sizi özledik' mesajları
                    - Son şans indirimleri
                    - Ürün iade/değişim kolaylığı
                    """
                }
                
                for segment, strategy in segments_strategies.items():
                    with st.expander(f"{segment} için Strateji"):
                        st.markdown(strategy)
            
            except Exception as e:
                st.error(f"Müşteri segmentasyonu yapılırken bir hata oluştu: {str(e)}")
                st.info("""
                Müşteri analizi için en az şu sütunlar gereklidir:
                - musteri_id: Müşteri kimlik numarası
                - tarih: Satış tarihi
                - satis_tutari: Satış tutarı
                
                Veri formatınızı kontrol edin ve yeniden deneyin.
                """)
        
        # Ürün Analizi
        analyze_products(filtered_df)
        
        # Tahminleme
        st.header("🔮 Satış Tahmini")
        
        forecast_days = st.slider("Tahmin Gün Sayısı", 7, 90, 30)
        
        if st.button("Tahmin Oluştur"):
            with st.spinner("Tahmin hesaplanıyor..."):
                forecast_results = forecast_sales(filtered_df, forecast_days)
                
                # Prophet tahmin grafiği
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast_results['prophet_forecast']['ds'],
                    y=forecast_results['prophet_forecast']['yhat'],
                    name='Prophet Tahmini',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_results['prophet_forecast']['ds'],
                    y=forecast_results['prophet_forecast']['yhat_lower'],
                    name='Alt Sınır',
                    line=dict(color='gray', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_results['prophet_forecast']['ds'],
                    y=forecast_results['prophet_forecast']['yhat_upper'],
                    name='Üst Sınır',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty'
                ))
                
                fig.update_layout(
                    title='Satış Tahmini',
                    xaxis_title='Tarih',
                    yaxis_title='Tahmini Satış Tutarı (₺)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ARIMA tahmin grafiği
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd.date_range(start=forecast_results['last_date'], periods=forecast_days),
                    y=forecast_results['arima_forecast'],
                    name='ARIMA Tahmini',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title='ARIMA Model Tahmini',
                    xaxis_title='Tarih',
                    yaxis_title='Tahmini Satış Tutarı (₺)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Stok Optimizasyonu
        st.header("📦 Stok Optimizasyonu")
        
        if st.button("Stok Analizi Oluştur"):
            with st.spinner("Stok analizi hesaplanıyor..."):
                stock_analysis = optimize_stock(filtered_df)
                
                st.subheader("Stok Durumu ve Öneriler")
                st.dataframe(stock_analysis)
                
                # Stok durumu dağılımı
                stock_status = stock_analysis['Stok Durumu'].value_counts()
                fig = px.pie(values=stock_status.values,
                           names=stock_status.index,
                           title='Stok Durumu Dağılımı')
                st.plotly_chart(fig, use_container_width=True)
        
        # Rapor Oluşturma
        st.header("📄 Kapsamlı Rapor")
        
        if st.button("Rapor Oluştur", key="create_report"):
            with st.spinner("Kapsamlı rapor oluşturuluyor..."):
                # Tahmin ve stok analizi sonuçlarını al
                forecast_results = forecast_sales(filtered_df)
                stock_analysis = optimize_stock(filtered_df)
                
                # Rapor oluştur
                report_html = generate_report(filtered_df, forecast_results, stock_analysis)
                
                # Tam sayfa rapor görüntüleme
                st.subheader("📊 E-Ticaret Satış Analiz Raporu")
                st.info("Aşağıda oluşturulan kapsamlı raporu görüntüleyebilirsiniz. Rapor interaktif olup, tablolar arasında geçiş yapabilir ve detaylı analizleri inceleyebilirsiniz.")
                
                # HTML'i göster - tam boy (yüksekliği artırdık ve kaydırma özelliğini ekledik)
                st.components.v1.html(report_html, height=1500, scrolling=True)
                
                # PDF indirme linki
                st.download_button(
                    label="📥 Raporu HTML Olarak İndir",
                    data=report_html.encode(),
                    file_name=f"e_ticaret_raporu_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
                    mime="text/html"
                )
        
        # Ham veri görüntüleme
        with st.expander("Ham Veri"):
            st.dataframe(filtered_df)
            
else:
    st.info("""
    Analiz yapmak için lütfen bir veri dosyası yükleyin.
    
    Beklenen veri formatı:
    - tarih: Satış tarihi
    - urun_adi: Ürün adı
    - miktar: Satış miktarı
    - satis_tutari: Satış tutarı (TL)
    - musteri_id: Müşteri ID (opsiyonel, RFM analizi için)
    - siparis_id: Sipariş ID (opsiyonel, RFM analizi için)
    - kategori: Ürün kategorisi (opsiyonel, kategori analizi için)
    """)

