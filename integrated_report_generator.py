"""
통합 DISE 분석 보고서 생성기 (Professional Medical Dashboard Design)
"""

from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import shutil
import google.generativeai as genai
import matplotlib.ticker as ticker
import numpy as np

class IntegratedReportGenerator:
    def __init__(self, results, api_key=None):
        self.results = results
        self.video_info = results.get('video_info', {})
        self.segments = results.get('segments', [])
        self.events = results.get('occlusion_events', [])
        self.summary = results.get('summary', {})
        self.patient_info = results.get('patient_info', {})
        self.api_key = api_key
        self.segment_references = results.get('segment_references', {})
        self.reference_images = results.get('reference_images', {})
        self.threshold_percent = results.get('metadata', {}).get('threshold_percent', 30)

    # ===================== VQA: 컨텍스트 & 질의응답 =====================
    def build_analysis_context(self):
        """VQA용 분석 컨텍스트 텍스트 생성"""
        segment_info = ""
        for label, ref_data in self.segment_references.items():
            segment_info += f"- {label} 영역 기준 면적: {ref_data['max_area']:.0f} px² (Frame {ref_data['frame_number']})\n"

        events_detail = ""
        for i, event in enumerate(self.events, 1):
            events_detail += f"""
이벤트 #{i}:
  - 부위: {event['segment_label']}
  - 등급: {event['severity']}
  - 시간: {event['start_time']:.1f}s ~ {event['end_time']:.1f}s (지속시간: {event['duration']:.1f}s)
  - 최대 감소율: {event.get('max_reduction', 0):.1f}%
  - 기준 면적: {event.get('segment_max_area', 0):.0f} px²
"""

        context = f"""
[환자 기본 정보]
- 성별/나이: {self.patient_info.get('gender','미상')} / {self.patient_info.get('age','미상')}세
- 기저 질환: {self.patient_info.get('diag','미상')}
- AHI: {self.patient_info.get('AHI','미상')}

[영상 정보]
- 파일명: {self.video_info.get('filename', '미상')}
- 영상 길이: {self.video_info.get('duration', 0):.1f}초

[해부학적 부위별 기준 면적]
{segment_info or '정보 없음'}

[분석 요약]
- 감지된 구간: OTE {self.summary.get('ote_segments',0)}개, Velum {self.summary.get('velum_segments',0)}개
- 전체 폐색 이벤트: {self.summary.get('total_events',0)}개
- 등급별 분포:
  * Grade 2 (Complete, >75%): {self.summary.get('events_by_severity',{}).get('Grade 2',0)}개
  * Grade 1 (Partial, 50-75%): {self.summary.get('events_by_severity',{}).get('Grade 1',0)}개
  * Grade 0 (None, <50%): {self.summary.get('events_by_severity',{}).get('Grade 0',0)}개

[Visual Occlusion 등급 기준]
- Grade 0: 폐쇄 없음 (< 50%)
- Grade 1: 부분 폐쇄 (50% ~ 75%)
- Grade 2: 완전 폐쇄 (> 75%)

[감지된 폐색 이벤트 상세]
{events_detail or '폐색 이벤트가 감지되지 않았습니다.'}
"""
        return context

    def answer_question(self, question: str, conversation_history=None):
        if not self.api_key: return {"success": False, "error": "API Key Missing"}
        try:
            genai.configure(api_key=self.api_key.strip())
            context = self.build_analysis_context()
            system_prompt = f"""
[역할] 수면 무호흡증(OSA) 전문의.
[분석 데이터]
{context}
[지침]
1. Visual Occlusion Grading System (Grade 0/1/2)에 기반하여 한국어로 답변하세요.
2. Grade 2(완전 폐쇄) 이벤트가 있다면 중점적으로 설명하세요.
3. 데이터에 기반하여 구체적 수치를 언급하세요.
"""
            if conversation_history:
                history = []
                for msg in conversation_history:
                    role = "user" if msg['role'] == 'user' else "model"
                    history.append({"role": role, "parts": [msg['content']]})
                model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_prompt)
                chat = model.start_chat(history=history)
                response = chat.send_message(question)
            else:
                model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_prompt)
                response = model.generate_content(question)
            return {"success": True, "answer": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_ai_summary(self):
        if not self.api_key: return "API Key Not Found."
        try:
            genai.configure(api_key=self.api_key)
            context = self.build_analysis_context()
            prompt = f"""
            [역할] 이비인후과 전문의
            [데이터]
            {context}
            [요청]
            위 데이터를 바탕으로 '의료진용 DISE 판독 소견서'를 작성하세요.
            1. 환자 개요 및 중증도 평가
            2. 부위별(OTE/Velum) 폐색 패턴 분석 (Grade 0~2 기준 사용)
            3. Grade 2(완전 폐쇄) 발생 빈도 및 지속시간에 대한 임상적 해석
            4. 종합 평가 및 치료 제안
            """
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text.replace('\n', '<br>')
        except: return "AI Analysis Failed."

    def generate_chart_interpretation(self, chart_type):
        if not self.api_key: return "AI 해석 불가"
        try:
            genai.configure(api_key=self.api_key)
            context = self.build_analysis_context()
            if chart_type == 'timeline':
                prompt = f"타임라인 차트 데이터를 보고 환자의 시간에 따른 폐쇄 패턴(Grade 변화)을 한 문장으로 요약해줘.\n{context}"
            else:
                prompt = f"등급별(Grade 0/1/2) 통계를 보고 전체적인 폐쇄 심각도를 한 문장으로 평가해줘.\n{context}"
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text.replace('\n', '<br>')
        except: return "해석 실패"

    # ===================== 차트 디자인 업그레이드 =====================
    def generate_timeline_chart(self, output_dir):
        # 스타일 설정
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
        fig.patch.set_facecolor('white')
        
        frames = self.results.get('frame_classifications', [])
        times = [f['timestamp'] for f in frames]
        areas = [f['roi_area'] for f in frames]
        
        # Main Area Chart
        ax1.plot(times, areas, color='#3b82f6', linewidth=2, label='Airway Area')
        ax1.fill_between(times, areas, color='#3b82f6', alpha=0.1)
        
        # Events Marking
        for event in self.events:
            color = '#ef4444' if event['severity'] == 'Grade 2' else '#f97316' if event['severity'] == 'Grade 1' else '#10b981'
            ax1.axvspan(event['start_time'], event['end_time'], color=color, alpha=0.15, zorder=1)
            ax1.hlines(y=max(areas)*0.95, xmin=event['start_time'], xmax=event['end_time'], color=color, linewidth=4)
            
        ax1.set_ylabel('Airway Area (px²)', fontsize=12, fontweight='bold', color='#475569')
        ax1.set_title(f'Dynamic Airway Analysis (Threshold: {self.threshold_percent}%)', fontsize=16, fontweight='bold', pad=20, color='#1e293b')
        ax1.grid(True, alpha=0.2, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Anatomy Bars (Gantt style)
        colors = {'OTE': '#3b82f6', 'Velum': '#8b5cf6'} # Blue & Purple
        for label, y_pos in [('Velum', 0), ('OTE', 1)]:
            segs = [s for s in self.segments if s['label'] == label]
            for seg in segs:
                ax2.barh(y_pos, seg['duration'], left=seg['start_time'], height=0.5, color=colors[label], edgecolor='white', alpha=0.9)
                if seg['duration'] > 1.0:
                    ax2.text(seg['start_time']+seg['duration']/2, y_pos, label, ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Velum', 'OTE'], fontsize=11, fontweight='bold', color='#475569')
        ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold', color='#475569')
        ax2.set_ylim(-0.5, 1.5)
        ax2.grid(axis='x', alpha=0.2, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        plt.subplots_adjust(hspace=0.15)
        plt.savefig(Path(output_dir)/'timeline.png', bbox_inches='tight', dpi=150)
        plt.close()

    def generate_severity_chart(self, output_dir):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')

        # 이벤트에서 Open/Partial/Complete 재집계 (Grade 0/1/2 매핑)
        grade_counts = {'Grade 0': 0, 'Grade 1': 0, 'Grade 2': 0}
        for ev in (self.events or []):
            sev = (ev.get('severity') or '').lower()
            if 'grade 2' in sev or 'complete' in sev or 'critical' in sev or 'full' in sev:
                grade_counts['Grade 2'] += 1
            elif 'grade 1' in sev or 'partial' in sev or 'severe' in sev:
                grade_counts['Grade 1'] += 1
            else:
                grade_counts['Grade 0'] += 1

        labels = ['Open', 'Partial', 'Complete']
        values = [grade_counts['Grade 0'], grade_counts['Grade 1'], grade_counts['Grade 2']]
        
        # Colors: Green, Orange, Red
        colors = ['#10b981', '#f97316', '#ef4444'] 
        
        bars = ax.bar(labels, values, color=colors, alpha=0.9, width=0.6, edgecolor='white', linewidth=2)
        
        ax.set_title('Occlusion Status Distribution (Open / Partial / Complete)', fontsize=16, fontweight='bold', pad=20, color='#1e293b')
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', labelsize=12, colors='#475569')
        ax.tick_params(axis='y', labelsize=10, colors='#94a3b8')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                        str(int(height)), ha='center', va='bottom', 
                        fontsize=14, fontweight='bold', color='#334155')
        
        plt.savefig(Path(output_dir)/'severity_chart.png', bbox_inches='tight', dpi=150)
        plt.close()

    # ===================== HTML 리포트 생성 (Modern UI) =====================
    def generate_html_report(self, output_dir):
        output_dir = Path(output_dir)
        self.generate_timeline_chart(output_dir)
        self.generate_severity_chart(output_dir)
        ai_note = self.generate_ai_summary()
        timeline_ai = self.generate_chart_interpretation('timeline')
        severity_ai = self.generate_chart_interpretation('severity')
        
        # [Date Logic] 메타데이터 우선, 없으면 오늘 날짜
        p_info = self.patient_info
        # 메타데이터 키 후보들을 확인하여 날짜 추출
        exam_date = p_info.get('date') or p_info.get('study_date') or p_info.get('exam_date')
        if not exam_date:
            exam_date = datetime.now().strftime('%Y-%m-%d')
        
        # --- Reference Images HTML ---
        ref_images_html = ""
        # 이미지 더 크게 표시 (manual: h-80, auto: h-64)
        if self.results.get('manual_ref_image'):
            web_path = f"overlays/{Path(self.results['manual_ref_image']).name}"
            ref_images_html = f"""
            <div class="relative group rounded-2xl overflow-hidden border border-slate-200 shadow-sm transition-all hover:shadow-md">
                <div class="absolute top-3 left-3 bg-indigo-600 text-white text-[10px] font-bold px-2.5 py-1 rounded-full shadow-sm z-10">Manual Ref</div>
                <div class="w-full aspect-[4/3] bg-slate-100">
                    <img src="{web_path}" class="w-full h-full object-cover"> 
                </div>
                <div class="bg-white p-3 border-t border-slate-50 flex justify-between items-center">
                    <span class="text-xs font-semibold text-slate-500">Max Area</span>
                    <span class="text-sm font-bold text-slate-800">{self.results.get('max_area',0):.0f} px²</span>
                </div>
            </div>"""
        elif self.reference_images:
            # 한 줄(1컬럼)로 배치해 좌우폭을 넓힘
            ref_images_html = "<div class='grid grid-cols-1 gap-4'>"
            for label in ['OTE', 'Velum']:
                if label in self.reference_images:
                    web_path = f"overlays/{Path(self.reference_images[label]).name}"
                    color_cls = "bg-sky-500" if label == "OTE" else "bg-purple-500"
                    ref_data = self.segment_references.get(label, {})
                    ref_images_html += f"""
                    <div class="relative group rounded-2xl overflow-hidden border border-slate-200 shadow-sm transition-all hover:shadow-md">
                        <div class="absolute top-3 left-3 {color_cls} text-white text-[10px] font-bold px-2.5 py-1 rounded-full shadow-sm z-10">{label}</div>
                        <div class="w-full aspect-[4/3] bg-slate-100">
                            <img src="{web_path}" class="w-full h-full object-cover">
                        </div>
                        <div class="bg-white p-2 border-t border-slate-50 text-right">
                             <span class="text-xs font-bold text-slate-700">{ref_data.get('max_area', 0):.0f} px²</span>
                        </div>
                    </div>"""
            ref_images_html += "</div>"
        else:
            ref_images_html = "<div class='p-8 text-center text-slate-400 bg-slate-50 rounded-2xl border border-dashed border-slate-200'>No Reference Images</div>"
        
        # --- Calculate Top Stats ---
        # 이벤트 리스트 기반으로 Grade 0/1/2를 직접 집계 (summary 신뢰도 보완)
        total_events = len(self.events or [])
        grade_counts = {'Grade 0': 0, 'Grade 1': 0, 'Grade 2': 0}
        for ev in (self.events or []):
            sev = (ev.get('severity') or '').strip()
            sev_lower = sev.lower()
            if sev in grade_counts:
                grade_counts[sev] += 1
            elif 'complete' in sev_lower or 'grade 2' in sev_lower or 'critical' in sev_lower or 'full' in sev_lower:
                grade_counts['Grade 2'] += 1
            elif 'partial' in sev_lower or 'grade 1' in sev_lower or 'severe' in sev_lower:
                grade_counts['Grade 1'] += 1
            else:
                grade_counts['Grade 0'] += 1

        count_g2 = grade_counts['Grade 2']
        count_g1 = grade_counts['Grade 1']
        count_g0 = grade_counts['Grade 0']
        
        dominant_region = "None"
        if self.summary.get('ote_segments', 0) > self.summary.get('velum_segments', 0): dominant_region = "OTE"
        elif self.summary.get('velum_segments', 0) > 0: dominant_region = "Velum"
        
        video_stem = Path(self.video_info.get('filename','')).stem

        # --- HTML Structure ---
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DISE Clinical Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Manrope', sans-serif; background: #f8fafc; color: #1e293b; }}
        .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
        
        /* Modern Card Utility */
        .card {{ background: white; border-radius: 1.5rem; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02); transition: transform 0.2s; }}
        .card:hover {{ box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025); }}
        
        /* Grade Badges */
        .badge {{ padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }}
        .badge-g2 {{ background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }}
        .badge-g1 {{ background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }}
        .badge-g0 {{ background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }}
        
        /* Row Hover Colors for Table */
        .hover-row-g2:hover {{ background-color: #fef2f2; }}
        .hover-row-g1:hover {{ background-color: #fff7ed; }}
        .hover-row-g0:hover {{ background-color: #f0fdf4; }}

        /* Layout Utilities */
        .glass-header {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-bottom: 1px solid #e2e8f0; }}
        
        /* Chat Animation */
        .chat-slide-enter {{ animation: slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards; }}
        .chat-slide-exit {{ animation: slideDown 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards; }}
        @keyframes slideUp {{ from {{ opacity: 0; transform: translateY(20px) scale(0.96); }} to {{ opacity: 1; transform: translateY(0) scale(1); }} }}
        @keyframes slideDown {{ from {{ opacity: 1; transform: translateY(0) scale(1); }} to {{ opacity: 0; transform: translateY(20px) scale(0.96); }} }}
        
        .no-scrollbar::-webkit-scrollbar {{ display: none; }}
    </style>
</head>
<body class="selection:bg-indigo-100 selection:text-indigo-900">

    <nav class="fixed top-0 w-full z-40 glass-header h-16 flex items-center px-6 lg:px-12 justify-between">
        <div class="flex items-center gap-3">
            <div class="w-8 h-8 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center text-white shadow-md">
                <i class="fas fa-wave-square text-xs"></i>
            </div>
            <span class="font-bold text-lg tracking-tight text-slate-800">DISE <span class="text-indigo-600">Analytics</span></span>
        </div>
        <div class="flex items-center gap-6">
            <div class="text-right">
                <p class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Exam Date</p>
                <p class="text-sm font-bold text-slate-900">{exam_date}</p>
            </div>
            <div class="h-8 w-px bg-slate-200"></div>
            <div class="text-right">
                <p class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Patient ID</p>
                <p class="text-sm font-bold text-slate-900">{p_info.get('id', 'UNKNOWN')}</p>
            </div>
        </div>
    </nav>

    <main class="pt-24 pb-20 px-6 lg:px-12 max-w-[1600px] mx-auto space-y-8">
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div class="card p-6 flex flex-col justify-between relative overflow-hidden group">
                <div class="absolute right-0 top-0 p-4 opacity-10 group-hover:scale-110 transition-transform"><i class="fas fa-user-circle text-6xl text-indigo-500"></i></div>
                <div>
                    <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Patient</p>
                    <h3 class="text-xl font-bold text-slate-800 mt-1">{p_info.get('age','-')} <span class="text-sm font-normal text-slate-500">Years</span> / {p_info.get('gender','-')}</h3>
                </div>
                <div class="mt-4 pt-4 border-t border-slate-100">
                    <p class="text-xs text-slate-500">Diagnosis</p>
                    <p class="text-sm font-bold text-indigo-600 truncate">{p_info.get('diag','Unknown')}</p>
                </div>
            </div>

            <div class="card p-6 flex flex-col justify-between relative overflow-hidden group">
                <div class="absolute right-0 top-0 p-4 opacity-10 group-hover:scale-110 transition-transform"><i class="fas fa-exclamation-triangle text-6xl text-rose-500"></i></div>
                <div>
                    <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Total Events</p>
                    <h3 class="text-4xl font-extrabold text-slate-900 mt-1 font-mono">{total_events}</h3>
                </div>
                <div class="mt-4 flex gap-2">
                    <span class="px-2 py-1 rounded bg-red-50 text-red-700 text-xs font-bold border border-red-100">{count_g2} Complete</span>
                    <span class="px-2 py-1 rounded bg-orange-50 text-orange-700 text-xs font-bold border border-orange-100">{count_g1} Partial</span>
                    <span class="px-2 py-1 rounded bg-green-50 text-green-700 text-xs font-bold border border-green-100">{count_g0} Open</span>
                </div>
            </div>

             <div class="card p-6 flex flex-col justify-between relative overflow-hidden group">
                <div class="absolute right-0 top-0 p-4 opacity-10 group-hover:scale-110 transition-transform"><i class="fas fa-lungs text-6xl text-emerald-500"></i></div>
                <div>
                    <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">AHI Score</p>
                    <h3 class="text-4xl font-extrabold text-slate-900 mt-1 font-mono">{p_info.get('AHI', '-')}</h3>
                </div>
                 <div class="mt-4 pt-4 border-t border-slate-100">
                    <p class="text-xs text-slate-500">Reference Max Area</p>
                    <p class="text-sm font-bold text-slate-700">{self.results.get('max_area',0):.0f} px²</p>
                </div>
            </div>
            
            <div class="card p-6 flex flex-col justify-between relative overflow-hidden group">
                <div class="absolute right-0 top-0 p-4 opacity-10 group-hover:scale-110 transition-transform"><i class="fas fa-chart-area text-6xl text-purple-500"></i></div>
                <div>
                    <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Dominant Region</p>
                    <h3 class="text-3xl font-bold text-slate-900 mt-1">{dominant_region}</h3>
                </div>
                 <div class="mt-4 pt-4 border-t border-slate-100">
                    <p class="text-xs text-slate-500">Video Duration</p>
                    <p class="text-sm font-bold text-slate-700">{self.video_info.get('duration', 0):.1f} sec</p>
                </div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-2 card p-8 border-l-4 border-l-indigo-600 bg-gradient-to-br from-white to-slate-50">
                <div class="flex items-center gap-3 mb-6">
                    <div class="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600">
                        <i class="fas fa-robot text-lg"></i>
                    </div>
                    <div>
                        <h2 class="text-lg font-bold text-slate-800">AI Clinical Insight</h2>
                        <p class="text-xs text-slate-500">Automated analysis summary based on visual occlusion grading</p>
                    </div>
                </div>
                <div class="prose prose-sm max-w-none text-slate-700 leading-relaxed">
                    {ai_note}
                </div>
            </div>

            <div class="lg:col-span-1 flex flex-col gap-4">
                <div class="card p-6 h-full">
                    <h3 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Reference Anatomy</h3>
                    {ref_images_html}
                </div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-2 card p-6">
                <div class="flex justify-between items-center mb-6">
                    <h3 class="font-bold text-slate-800 text-lg">Timeline Analysis</h3>
                    <span class="text-xs font-mono bg-slate-100 px-2 py-1 rounded text-slate-500">Interactive Plot</span>
                </div>
                <img src="timeline.png" class="w-full rounded-xl border border-slate-100 shadow-sm">
                <div class="mt-4 p-4 bg-slate-50 rounded-xl border border-slate-100 flex gap-3 items-start">
                    <i class="fas fa-info-circle text-indigo-500 mt-0.5"></i>
                    <p class="text-xs text-slate-600 leading-relaxed"><strong>Interpretation:</strong> {timeline_ai}</p>
                </div>
            </div>

            <div class="lg:col-span-1 card p-6 flex flex-col">
                <h3 class="font-bold text-slate-800 text-lg mb-6">Severity Distribution</h3>
                <div class="flex-1 flex items-center justify-center p-4 bg-slate-50 rounded-xl border border-slate-100 border-dashed">
                    <img src="severity_chart.png" class="max-h-[250px] object-contain">
                </div>
                 <div class="mt-4 p-4 bg-slate-50 rounded-xl border border-slate-100 flex gap-3 items-start">
                    <i class="fas fa-chart-pie text-orange-500 mt-0.5"></i>
                    <p class="text-xs text-slate-600 leading-relaxed"><strong>Interpretation:</strong> {severity_ai}</p>
                </div>
            </div>
        </div>

        <div class="card overflow-hidden">
            <div class="px-8 py-6 border-b border-slate-100 flex justify-between items-center bg-white">
                <div class="flex items-center gap-3">
                    <div class="w-8 h-8 rounded-full bg-rose-50 flex items-center justify-center text-rose-500">
                        <i class="fas fa-list-ul"></i>
                    </div>
                    <h3 class="font-bold text-lg text-slate-800">Event Logs</h3>
                </div>
                <div class="flex gap-2">
                     <span class="px-3 py-1 bg-slate-100 rounded-full text-xs font-bold text-slate-600 border border-slate-200">{len(self.events)} Events Detected</span>
                </div>
            </div>
            
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="bg-slate-50/50 border-b border-slate-100 text-xs uppercase text-slate-400 font-bold tracking-wider">
                            <th class="px-8 py-4">Severity Grade</th>
                            <th class="px-6 py-4">Anatomy</th>
                            <th class="px-6 py-4">Timestamp</th>
                            <th class="px-6 py-4">Duration</th>
                            <th class="px-6 py-4">Max Reduction</th>
                            <th class="px-6 py-4 text-center">Replay</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-50 text-sm">
        """
        
        if not self.events:
             html += '<tr><td colspan="6" class="px-8 py-12 text-center text-slate-400 italic">No significant occlusion events detected.</td></tr>'
        else:
            for i, event in enumerate(self.events):
                clip_path = f"event_clips/{Path(event.get('clip_path','')).name}"
                
                # [수정] Grade별 Badge 및 Row Hover 색상 설정
                severity = event['severity']
                # Grade를 Open/Partial/Complete로 매핑
                severity_display = severity
                if severity == 'Grade 2':
                    severity_display = 'Complete'
                    badge_class = 'badge-g2'
                    row_hover = 'hover-row-g2'
                elif severity == 'Grade 1':
                    severity_display = 'Partial'
                    badge_class = 'badge-g1'
                    row_hover = 'hover-row-g1'
                elif severity == 'Grade 0':
                    severity_display = 'Open'
                    badge_class = 'badge-g0'
                    row_hover = 'hover-row-g0'
                else:
                    badge_class = 'badge-g0'
                    row_hover = 'hover-row-g0'
                
                html += f"""
                <tr onclick="playVideo('{clip_path}', '{severity_display} - {event['segment_label']}')" class="{row_hover} transition-colors cursor-pointer group">
                    <td class="px-8 py-4">
                        <span class="badge {badge_class}">{severity_display}</span>
                    </td>
                    <td class="px-6 py-4 font-bold text-slate-700">{event['segment_label']}</td>
                    <td class="px-6 py-4 text-slate-500 font-mono text-xs">
                        <i class="far fa-clock mr-1 opacity-50"></i>{event['start_time']:.1f}s — {event['end_time']:.1f}s
                    </td>
                    <td class="px-6 py-4 text-slate-600 font-medium">{event['duration']:.1f}s</td>
                    <td class="px-6 py-4">
                        <div class="flex items-center gap-2">
                            <div class="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                <div class="h-full bg-rose-500" style="width: {event.get('max_reduction',0)}%"></div>
                            </div>
                            <span class="font-bold text-rose-600 text-xs">{event.get('max_reduction',0):.0f}%</span>
                        </div>
                    </td>
                    <td class="px-6 py-4 text-center">
                        <button class="w-8 h-8 rounded-full bg-white border border-slate-200 text-slate-400 group-hover:bg-indigo-600 group-hover:text-white group-hover:border-indigo-600 transition-all shadow-sm">
                            <i class="fas fa-play text-[10px] pl-0.5"></i>
                        </button>
                    </td>
                </tr>
                """
        
        # --- Chat & Modal HTML ---
        floating_chat_html = """
        <button id="chatToggleBtn" onclick="toggleChat()" class="fixed bottom-8 right-8 w-14 h-14 bg-indigo-600 text-white rounded-full shadow-[0_8px_30px_rgb(79,70,229,0.4)] flex items-center justify-center hover:scale-110 hover:bg-indigo-700 transition-all z-50 group">
            <i class="fas fa-comment-medical text-xl group-hover:animate-pulse"></i>
        </button>

        <div id="chatWindow" class="fixed bottom-24 right-8 w-[380px] h-[600px] bg-white rounded-2xl shadow-2xl border border-slate-200 flex flex-col z-50 hidden origin-bottom-right">
            <div class="p-4 border-b border-slate-100 flex justify-between items-center bg-white rounded-t-2xl">
                <div class="flex items-center gap-3">
                    <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center text-white text-xs shadow-md">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div>
                        <h4 class="font-bold text-sm text-slate-800">AI Medical Assistant</h4>
                        <p class="text-[10px] text-green-500 font-bold flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-green-500"></span> Online</p>
                    </div>
                </div>
                <button onclick="toggleChat()" class="w-8 h-8 rounded-full hover:bg-slate-50 text-slate-400 flex items-center justify-center transition"><i class="fas fa-times"></i></button>
            </div>
            
            <div id="chatContainer" class="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/50">
                <div class="flex gap-3">
                    <div class="w-8 h-8 rounded-full bg-white border border-slate-200 flex items-center justify-center flex-shrink-0">
                        <i class="fas fa-robot text-indigo-500 text-xs"></i>
                    </div>
                    <div class="bg-white border border-slate-200 p-3 rounded-2xl rounded-tl-none text-sm text-slate-600 shadow-sm max-w-[85%]">
                        안녕하세요! 분석 결과에 대해 궁금한 점을 질문해주세요.
                    </div>
                </div>
            </div>
            
            <div class="p-4 bg-white border-t border-slate-100 rounded-b-2xl">
                <div class="flex gap-2 mb-3 overflow-x-auto no-scrollbar pb-1">
                    <button onclick="setQ('가장 심각한 이벤트 요약해줘')" class="whitespace-nowrap px-3 py-1.5 bg-indigo-50 text-indigo-700 text-xs font-bold rounded-full hover:bg-indigo-100 border border-indigo-100 transition">심각한 구간?</button>
                    <button onclick="setQ('치료 권고사항은?')" class="whitespace-nowrap px-3 py-1.5 bg-indigo-50 text-indigo-700 text-xs font-bold rounded-full hover:bg-indigo-100 border border-indigo-100 transition">치료 제안</button>
                    <button onclick="setQ('이벤트 횟수는 몇 번인가요?')" class="whitespace-nowrap px-3 py-1.5 bg-indigo-50 text-indigo-700 text-xs font-bold rounded-full hover:bg-indigo-100 border border-indigo-100 transition">이벤트 횟수</button>
                </div>
                <div class="relative">
                    <input type="text" id="vqaQuestion" class="w-full pl-4 pr-12 py-3 bg-slate-100 border-none rounded-xl text-sm focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all outline-none" placeholder="Type your question..." onkeypress="if(event.key==='Enter') askAI()">
                    <button onclick="askAI()" class="absolute right-2 top-2 w-8 h-8 bg-indigo-600 text-white rounded-lg flex items-center justify-center hover:bg-indigo-700 shadow-sm transition">
                        <i class="fas fa-paper-plane text-xs"></i>
                    </button>
                </div>
            </div>
        </div>
        """

        html += f"""
                    </tbody>
                </table>
            </div>
        </div>
        
        {floating_chat_html}

        <div id="videoModal" class="fixed inset-0 z-[100] hidden" onclick="closeModal()">
            <div class="fixed inset-0 bg-slate-900/80 backdrop-blur-sm transition-opacity"></div>
            <div class="fixed inset-0 flex items-center justify-center p-4">
                <div class="bg-black rounded-2xl overflow-hidden max-w-5xl w-full relative shadow-2xl border border-white/10" onclick="event.stopPropagation()">
                    <div class="bg-gradient-to-r from-slate-900 to-slate-800 px-6 py-4 flex justify-between items-center border-b border-white/10">
                        <h3 class="text-white font-bold flex items-center gap-2" id="modalTitle">
                            <i class="fas fa-play-circle text-indigo-400"></i> Event Replay
                        </h3>
                        <button onclick="closeModal()" class="w-8 h-8 rounded-full bg-white/10 text-white hover:bg-white/20 flex items-center justify-center transition"><i class="fas fa-times"></i></button>
                    </div>
                    <div class="aspect-video bg-black flex items-center justify-center">
                        <video id="player" controls class="w-full h-full"></video>
                    </div>
                </div>
            </div>
        </div>

    </main>

    <script>
        const videoStem = "{video_stem}";  // ✅ 이건 Python 변수니까 {{ 없이
        let history = [];
        let chatOpen = false;
        
        // typing indicator 스타일 추가
        (function() {{
            const style = document.createElement('style');
            // f-string 이스케이프: CSS 중괄호는 두 개씩 {{ }}
            style.textContent = `
                @keyframes blink {{
                    0%, 100% {{ opacity: 0.2; }}
                    50% {{ opacity: 1; }}
                }}
                .typing-dots span {{
                    display: inline-block;
                    width: 6px;
                    height: 6px;
                    margin-right: 4px;
                    background: #0ea5e9;
                    border-radius: 9999px;
                    animation: blink 1s infinite;
                }}
                .typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
                .typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
            `;
            document.head.appendChild(style);
        }})();
        
        function toggleChat() {{
            const win = document.getElementById('chatWindow');
            chatOpen = !chatOpen;
            if(chatOpen) {{
                win.classList.remove('hidden');
                win.classList.add('chat-slide-enter');
                win.classList.remove('chat-slide-exit');
                setTimeout(() => document.getElementById('vqaQuestion').focus(), 300);
            }} else {{
                win.classList.add('chat-slide-exit');
                win.classList.remove('chat-slide-enter');
                setTimeout(() => win.classList.add('hidden'), 280);
            }}
        }}
        
        function setQ(text) {{ 
            const input = document.getElementById('vqaQuestion');
            input.value = text;
            input.focus();
        }}
        
        async function askAI() {{
            const input = document.getElementById('vqaQuestion');
            const q = input.value.trim();
            if(!q) return;
            
            const cont = document.getElementById('chatContainer');
            // User Message (pulse 제거)
            cont.innerHTML += `
                <div class="flex items-end gap-2 justify-end">
                    <div class="bg-indigo-600 text-white p-3 rounded-2xl rounded-tr-none text-sm max-w-[85%] shadow-md">
                        ${{q}}
                    </div>
                </div>`;
            
            input.value = '';
            input.disabled = true;
            cont.scrollTop = cont.scrollHeight;
            
            history.push({{'role':'user', 'content':q}});
            
            // 타이핑 인디케이터 추가
            const loadingId = "typing-" + Date.now();
            cont.innerHTML += `
                <div id="${{loadingId}}" class="flex items-start gap-3 mt-2">
                    <div class="w-8 h-8 rounded-full bg-white border border-slate-200 flex items-center justify-center flex-shrink-0 shadow-sm">
                        <i class="fas fa-robot text-emerald-500 text-xs"></i>
                    </div>
                    <div class="bg-white border border-slate-200 p-3 rounded-2xl rounded-tl-none text-sm text-slate-700 shadow-sm max-w-[85%] leading-relaxed">
                        <div class="typing-dots flex items-center gap-1">
                            <span></span><span></span><span></span>
                        </div>
                    </div>
                </div>`;
            cont.scrollTop = cont.scrollHeight;
            
            try {{
                const res = await fetch('/api/vqa', {{
                    method:'POST', headers:{{'Content-Type':'application/json'}},
                    body: JSON.stringify({{question:q, video_stem:videoStem, conversation_history:history.slice(0,-1)}})
                }});
                const data = await res.json();
                const ans = data.success ? data.answer : "Error: " + data.error;
                
                // 타이핑 인디케이터 제거
                const loadingEl = document.getElementById(loadingId);
                if(loadingEl) loadingEl.remove();
                
                // Bot Message
                cont.innerHTML += `
                    <div class="flex items-start gap-3 mt-2">
                        <div class="w-8 h-8 rounded-full bg-white border border-slate-200 flex items-center justify-center flex-shrink-0 shadow-sm">
                            <i class="fas fa-robot text-emerald-500 text-xs"></i>
                        </div>
                        <div class="bg-white border border-slate-200 p-3 rounded-2xl rounded-tl-none text-sm text-slate-700 shadow-sm max-w-[85%] leading-relaxed">
                            ${{ans}}
                        </div>
                    </div>`;
                    
                history.push({{'role':'assistant', 'content':ans}});
            }} catch(e) {{
                cont.innerHTML += `<div class="text-center my-2"><span class="bg-red-50 text-red-500 text-xs px-2 py-1 rounded">Network Error</span></div>`;
            }} finally {{
                input.disabled = false;
                input.focus();
                cont.scrollTop = cont.scrollHeight;
            }}
        }}
        
        function playVideo(src, title) {{
            const m = document.getElementById('videoModal');
            const p = document.getElementById('player');
            const t = document.getElementById('modalTitle');
            
            p.src = src;
            t.innerHTML = `<i class="fas fa-play-circle text-indigo-400"></i> ${{title}}`;
            m.classList.remove('hidden');
            p.play();
        }}
        
        function closeModal() {{
            document.getElementById('videoModal').classList.add('hidden');
            const p = document.getElementById('player');
            p.pause();
            p.src = '';
        }}
    </script>
</body>
</html>"""
        
        with open(output_dir / 'report.html', 'w', encoding='utf-8') as f:
            f.write(html)
        return output_dir / 'report.html'

    def generate_report(self, output_dir):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        report_path = self.generate_html_report(output_dir_path)
        return report_path