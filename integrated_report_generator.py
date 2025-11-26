"""
통합 DISE 분석 보고서 생성기 (Segment-based References)
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

    def generate_ai_summary(self):
        if not self.api_key: return "API Key Not Found."
        try:
            genai.configure(api_key=self.api_key)
            
            # [수정 1] 의료 분석을 위해 안전 필터(Safety Settings)를 'BLOCK_NONE'으로 설정
            # 의료 텍스트가 '유해 콘텐츠'로 오분류되는 것을 방지합니다.
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            
            # 모델 생성 시 설정 적용
            model = genai.GenerativeModel('gemini-2.5-flash', safety_settings=safety_settings)
            # Segment별 기준 면적 정보 추가
            segment_info = ""
            for label, ref_data in self.segment_references.items():
                segment_info += f"- {label} 영역 기준 면적: {ref_data['max_area']:.0f} px² (Frame {ref_data['frame_number']})\n"
            
            prompt = f"""
            [역할] 수면 무호흡증(OSA) 진단 전문의
            
            [환자 기본 정보 (진료 기록)]
            - 성별/나이: {self.patient_info.get('gender','?')} / {self.patient_info.get('age','?')}세
            - 기저 질환 진단명: {self.patient_info.get('diag','-')}
            - AHI (수면다원검사 결과): {self.patient_info.get('AHI','-')}

            [이번 DISE 영상 분석 결과]
            - 분석된 영상 길이: {self.video_info.get('duration', 0):.1f}초
            - 감지된 폐색 이벤트: {self.summary.get('total_events',0)}회
            - 주요 부위별 구간 감지: OTE {self.summary.get('ote_segments',0)}구간, Velum {self.summary.get('velum_segments',0)}구간
            - 심각도 분포: {self.summary.get('events_by_severity',{})}
            
            [해부학적 부위별 기준 면적]
            {segment_info}
            
            [폐쇄 감지 방법]
            각 해부학적 부위(OTE/Velum)별로 해당 부위의 최대 기도 면적을 기준으로,
            기준 대비 {self.threshold_percent}% 이상 감소한 경우를 폐쇄 이벤트로 감지하였음.
            
            [작성 지침]
            위 데이터를 바탕으로 '의료진용 판독 소견서'를 한국어로 작성하시오. 다음 구조를 따르시오:
            1. **환자 개요:** 기저 정보(AHI 등)를 바탕으로 환자의 전반적인 중증도를 언급하시오.
            2. **영상 소견:** 부위별(OTE/Velum) 폐색 패턴을 기술하시오.
               - 각 부위별 기준 면적 대비 감소율로 평가
               - 이벤트가 0개라면 "해당 threshold 기준으로 특이적인 폐색이 관찰되지 않음"을 명시
            3. **종합 평가:** 추가 관찰이나 치료 필요성을 제안하시오.
            """
            resp = model.generate_content(prompt)
            return resp.text.replace('\n', '<br>')
        except: return "AI Analysis Failed."

    def generate_chart_interpretation(self, chart_type):
        if not self.api_key: return "AI 해석을 사용할 수 없습니다."
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            if chart_type == 'timeline':
                prompt = f"""
                [작업] 의사에게 이 타임라인 차트 데이터를 설명해주세요.
                [데이터]
                - 비디오 길이: {self.video_info.get('duration',0):.1f}초
                - 전체 이벤트 수: {len(self.events)}개
                - 구간 수: {len(self.segments)}개 (OTE/Velum)
                - 각 부위별 기준 면적: {self.segment_references}
                - Threshold: {self.threshold_percent}%
                [출력]
                환자의 시간에 따른 폐쇄 패턴을 한 문장의 한국어로 해석해주세요.
                """
            else:
                sev_dist = self.summary.get('events_by_severity', {})
                prompt = f"""
                [작업] 중증도 통계를 해석해주세요.
                [데이터] {sev_dist}
                [출력]
                전체 중증도 수준을 한 문장의 한국어로 평가해주세요.
                """
            resp = model.generate_content(prompt)
            return resp.text.replace('\n', '<br>')
        except: return "해석 생성 실패."

    def generate_timeline_chart(self, output_dir):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
        
        frames = self.results.get('frame_classifications', [])
        times = [f['timestamp'] for f in frames]
        areas = [f['roi_area'] for f in frames]
        
        # 1. Area Line
        ax1.plot(times, areas, color='#2c3e50', linewidth=2.5, label='Airway Area')
        ax1.fill_between(times, areas, color='#3498db', alpha=0.15)
        
        # Segment별 Threshold Line 그리기
        for segment in self.segments:
            if segment.get('max_area', 0) > 0:
                threshold_val = segment['max_area'] * (1 - self.threshold_percent / 100)
                color = '#3498db' if segment['label'] == 'OTE' else '#9b59b6'
                ax1.hlines(y=threshold_val, 
                          xmin=segment['start_time'], 
                          xmax=segment['end_time'],
                          color=color, linestyle='--', linewidth=2.5, alpha=0.8,
                          label=f'{segment["label"]} Threshold ({threshold_val:.0f})' if segment == self.segments[0] or segment['label'] != self.segments[0]['label'] else "")

        # Event 표시
        for event in self.events:
            ax1.axvspan(event['start_time'], event['end_time'], color='#e74c3c', alpha=0.25, zorder=1)
            mid_x = (event['start_time'] + event['end_time']) / 2
            max_area = max(areas) if areas else 1000
            
            ax1.text(mid_x, max_area * 0.85, 
                    '  EVENT  ', 
                    color='white', fontsize=16, fontweight='bold',
                    va='center', ha='center',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#c0392b', edgecolor='white', linewidth=2, alpha=0.95))

        ax1.set_ylabel('Airway Area (px²)', fontsize=14, fontweight='bold')
        ax1.set_title('ROI Area Change over Time (Label-based Thresholds)', fontsize=16, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        ax1.legend(loc='upper right', fontsize=12, framealpha=0.95)
        ax1.tick_params(labelsize=11)

        # 2. Anatomy Ribbon - 2줄로 분리 (OTE 위, Velum 아래)
        colors = {'OTE': '#3498db', 'Velum': '#9b59b6'}
        
        # OTE segments (y=1)
        ote_segments = [s for s in self.segments if s['label'] == 'OTE']
        for seg in ote_segments:
            ax2.barh(1, seg['duration'], left=seg['start_time'], height=0.4, 
                    color=colors['OTE'], edgecolor='white', linewidth=2, alpha=0.9)
            
            if seg['duration'] > 0.8:
                label_text = f"OTE"
                if seg.get('max_area'):
                    label_text += f"\n({seg['max_area']:.0f}px²)"
                
                ax2.text(seg['start_time'] + seg['duration']/2, 1, 
                        f'  {label_text}  ', 
                        ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['OTE'], 
                                edgecolor='white', linewidth=1.5, alpha=0.9))
        
        # Velum segments (y=0)
        velum_segments = [s for s in self.segments if s['label'] == 'Velum']
        for seg in velum_segments:
            ax2.barh(0, seg['duration'], left=seg['start_time'], height=0.4, 
                    color=colors['Velum'], edgecolor='white', linewidth=2, alpha=0.9)
            
            if seg['duration'] > 0.8:
                label_text = f"Velum"
                if seg.get('max_area'):
                    label_text += f"\n({seg['max_area']:.0f}px²)"
                
                ax2.text(seg['start_time'] + seg['duration']/2, 0, 
                        f'  {label_text}  ', 
                        ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['Velum'], 
                                edgecolor='white', linewidth=1.5, alpha=0.9))

        ax2.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Velum', 'OTE'], fontsize=11, fontweight='bold')
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_title('Anatomy Region (Separated by Label)', fontsize=14, fontweight='bold', pad=12)
        ax2.tick_params(labelsize=11)
        ax2.grid(axis='x', alpha=0.2, linestyle=':')
        
        plt.subplots_adjust(hspace=0.25)
        path = Path(output_dir) / 'timeline.png'
        plt.savefig(path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()

    def generate_severity_chart(self, output_dir):
        fig, ax = plt.subplots(figsize=(10, 6))
        severities = ['Mild', 'Moderate', 'Severe', 'Critical']
        counts = self.summary.get('events_by_severity', {})
        values = [counts.get(s, 0) for s in severities]
        colors = ['#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
        
        bars = ax.bar(severities, values, color=colors, alpha=0.8, width=0.6)
        ax.set_title('Event Severity Statistics', fontsize=16, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        str(int(bar.get_height())), ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        path = Path(output_dir) / 'severity_chart.png'
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()

    def generate_html_report(self, output_dir):
        output_dir = Path(output_dir)
        self.generate_timeline_chart(output_dir)
        self.generate_severity_chart(output_dir)
        ai_note = self.generate_ai_summary()
        timeline_ai = self.generate_chart_interpretation('timeline')
        severity_ai = self.generate_chart_interpretation('severity')
        
        # Reference 이미지들 HTML 생성
        ref_images_html = ""
        if self.reference_images:
            ref_images_html = "<div class='space-y-3'>"
            for label in ['OTE', 'Velum']:
                if label in self.reference_images:
                    img_path = Path(self.reference_images[label])
                    web_path = f"overlays/{img_path.name}"
                    ref_data = self.segment_references.get(label, {})
                    color = '#3498db' if label == 'OTE' else '#9b59b6'
                    
                    ref_images_html += f"""
                    <div class="relative group">
                        <div class="text-xs font-bold mb-1" style="color: {color}">{label} Reference (Max: {ref_data.get('max_area', 0):.0f} px²)</div>
                        <img src="{web_path}" class="w-full rounded-lg border-4 shadow-md transition-transform group-hover:scale-[1.02]" style="border-color: {color}">
                    </div>
                    """
            ref_images_html += "</div>"
        else:
            ref_images_html = "<div class='bg-gray-100 p-4 rounded text-center text-gray-500'>No Reference Images</div>"

        p_info = self.patient_info
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Medical Analysis Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; background: #f3f4f6; }}
        .card {{ background: white; padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e5e7eb; }}
        .severity-badge {{ padding: 0.25rem 0.75rem; border-radius: 99px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }}
        .s-Critical {{ background: #fee2e2; color: #991b1b; }} .s-Severe {{ background: #ffedd5; color: #9a3412; }}
        .s-Moderate {{ background: #fef3c7; color: #92400e; }} .s-Mild {{ background: #dcfce7; color: #166534; }}
        .ai-box {{ background: #f0f9ff; border-left: 4px solid #0ea5e9; padding: 1rem; border-radius: 0.5rem; font-size: 0.9rem; color: #0369a1; margin-top: 1rem; }}
    </style>
</head>
<body class="text-slate-800">
    <nav class="bg-slate-900 text-white h-16 flex items-center px-8 fixed w-full z-50 shadow-lg">
        <div class="flex items-center gap-3 font-bold text-xl"><i class="fas fa-heartbeat text-rose-500"></i> DISE AI Analytics (Segment-based)</div>
    </nav>

    <div class="pt-24 pb-12 px-8 max-w-7xl mx-auto space-y-8">
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="card lg:col-span-1 flex flex-col gap-6">
                <div>
                    <h3 class="text-sm font-bold text-slate-400 uppercase mb-4"><i class="fas fa-user"></i> Patient Info</h3>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between border-b pb-1"><span>ID</span><span class="font-bold">{p_info.get('id', '-')}</span></div>
                        <div class="flex justify-between border-b pb-1"><span>Age/Sex</span><span class="font-bold">{p_info.get('age', '-')} / {p_info.get('gender', '-')}</span></div>
                        <div class="flex justify-between border-b pb-1"><span>Diagnosis</span><span class="font-bold text-indigo-600">{p_info.get('diag', '-')}</span></div>
                        <div class="flex justify-between border-b pb-1"><span>Threshold</span><span class="font-bold text-red-600">{self.threshold_percent}%</span></div>
                    </div>
                </div>
                <div>
                    <h3 class="text-sm font-bold text-slate-400 uppercase mb-2">Reference Frames</h3>
                    {ref_images_html}
                </div>
            </div>

            <div class="card lg:col-span-2 border-t-4 border-t-indigo-500">
                <h3 class="text-lg font-bold text-indigo-700 mb-4 flex items-center gap-2">
                    <i class="fas fa-user-md"></i> AI Doctor's Note
                </h3>
                <div class="prose prose-sm max-w-none text-slate-700 leading-relaxed">
                    {ai_note}
                </div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="card lg:col-span-2">
                <h3 class="text-lg font-bold text-slate-700 mb-4">Timeline Analysis</h3>
                <img src="timeline.png" class="w-full rounded-lg border">
                <div class="ai-box">
                    <i class="fas fa-info-circle mr-2"></i> <strong>Chart Insight:</strong><br>
                    {timeline_ai}
                </div>
            </div>
            <div class="card lg:col-span-1">
                <h3 class="text-lg font-bold text-slate-700 mb-4">Severity Stats</h3>
                <img src="severity_chart.png" class="w-full rounded-lg border">
                <div class="ai-box">
                    <i class="fas fa-chart-bar mr-2"></i> <strong>Analysis:</strong><br>
                    {severity_ai}
                </div>
            </div>
        </div>

        <div class="card">
            <h3 class="text-lg font-bold text-slate-700 mb-6">Detected Events ({len(self.events)})</h3>
            <table class="w-full text-sm text-left">
                <thead class="bg-slate-50 text-slate-500 font-medium border-b">
                    <tr>
                        <th class="px-6 py-3">Severity</th>
                        <th class="px-6 py-3">Region</th>
                        <th class="px-6 py-3">Time</th>
                        <th class="px-6 py-3">Reduction</th>
                        <th class="px-6 py-3">Ref Area</th>
                        <th class="px-6 py-3 text-center">Play</th>
                    </tr>
                </thead>
                <tbody class="divide-y">
        """
        
        if not self.events:
            html += '<tr><td colspan="6" class="px-6 py-8 text-center text-slate-400">No events detected.</td></tr>'
        else:
            for i, event in enumerate(self.events):
                clip_file = Path(event.get('clip_path', '')).name
                video_path = f"event_clips/{clip_file}"
                ref_area = event.get('segment_max_area', 0)
                html += f"""
                    <tr onclick="playVideo('{video_path}', 'Event #{i+1}')" class="hover:bg-slate-50 cursor-pointer transition">
                        <td class="px-6 py-4"><span class="severity-badge s-{event['severity']}">{event['severity']}</span></td>
                        <td class="px-6 py-4 font-bold text-slate-700">{event['segment_label']}</td>
                        <td class="px-6 py-4 text-slate-500">{event['start_time']:.1f}s ~ {event['end_time']:.1f}s</td>
                        <td class="px-6 py-4 font-bold text-red-600">{event['max_reduction']:.1f}%</td>
                        <td class="px-6 py-4 text-slate-600">{ref_area:.0f} px²</td>
                        <td class="px-6 py-4 text-center">
                            <button class="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 hover:bg-indigo-600 hover:text-white transition">
                                <i class="fas fa-play text-xs"></i>
                            </button>
                        </td>
                    </tr>
                """

        html += """
                </tbody>
            </table>
        </div>
    </div>

    <div id="videoModal" class="fixed inset-0 z-[100] hidden" onclick="closeModal()">
        <div class="fixed inset-0 bg-slate-900/90 backdrop-blur-sm"></div>
        <div class="fixed inset-0 flex items-center justify-center p-4">
            <div class="bg-black rounded-2xl shadow-2xl overflow-hidden max-w-7xl w-full relative" onclick="event.stopPropagation()">
                <div class="bg-slate-800 px-4 py-3 flex justify-between items-center">
                    <h3 class="text-white font-bold" id="modalTitle">Event Video (좌측: 원본, 우측: 분석 결과)</h3>
                    <button onclick="closeModal()" class="text-slate-400 hover:text-white"><i class="fas fa-times text-xl"></i></button>
                </div>
                <div class="bg-black flex items-center justify-center" style="min-height: 400px;">
                    <video id="player" controls class="w-full h-auto max-h-[80vh]" style="object-fit: contain;"></video>
                </div>
            </div>
        </div>
    </div>

    <script>
        function playVideo(src, title) {{
            const player = document.getElementById('player');
            player.innerHTML = '';
            player.onerror = null;
            player.onloadeddata = null;
            
            const source = document.createElement('source');
            source.src = src;
            source.type = 'video/mp4';
            player.appendChild(source);
            
            player.onerror = function(e) {{
                console.error('비디오 로드 실패:', src, e);
                alert('비디오 파일을 재생할 수 없습니다.\\n경로: ' + src);
            }};
            
            document.getElementById('modalTitle').innerText = title + ' (좌측: 원본, 우측: 분석 결과)';
            document.getElementById('videoModal').classList.remove('hidden');
            
            player.load();
            const playPromise = player.play();
            if (playPromise !== undefined) {{
                playPromise.catch(err => console.log('자동 재생 실패:', err));
            }}
        }}
        function closeModal() {{
            document.getElementById('videoModal').classList.add('hidden');
            const player = document.getElementById('player');
            player.pause();
            player.currentTime = 0;
            player.innerHTML = '';
        }}
        document.addEventListener('keydown', (e) => {{ if(e.key === 'Escape') closeModal(); }});
    </script>
</body>
</html>
        """
        
        with open(output_dir / 'report.html', 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_dir / 'report.html'

    def generate_report(self, output_dir):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        report_path = self.generate_html_report(output_dir_path)
        return report_path