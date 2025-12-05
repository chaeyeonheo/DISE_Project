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
  - 심각도: {event['severity']}
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
- FPS: {self.video_info.get('fps', 0):.1f}

[해부학적 부위별 기준 면적]
{segment_info or '정보 없음'}

[분석 요약]
- 감지된 구간: OTE {self.summary.get('ote_segments',0)}개, Velum {self.summary.get('velum_segments',0)}개
- 전체 폐색 이벤트: {self.summary.get('total_events',0)}개
- 심각도 분포:
  * Critical: {self.summary.get('events_by_severity',{}).get('Critical',0)}개
  * Severe: {self.summary.get('events_by_severity',{}).get('Severe',0)}개
  * Moderate: {self.summary.get('events_by_severity',{}).get('Moderate',0)}개
  * Mild: {self.summary.get('events_by_severity',{}).get('Mild',0)}개

[폐색 감지 방법]
각 해부학적 부위(OTE/Velum)별로 해당 부위의 최대 기도 면적을 기준으로,
기준 대비 {self.threshold_percent}% 이상 감소한 경우를 폐쇄 이벤트로 감지.

[감지된 폐색 이벤트 상세]
{events_detail or '폐색 이벤트가 감지되지 않았습니다.'}
"""
        return context

    def answer_question(self, question: str, conversation_history=None):
        """VQA: 분석 결과 기반 자연어 질의응답 (Multi-turn 지원)"""
        if not self.api_key:
            return {"success": False, "error": "Gemini API Key가 설정되지 않았습니다."}

        try:
            genai.configure(api_key=self.api_key)

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            context = self.build_analysis_context()

            # 시스템 프롬프트
            system_prompt = f"""
[역할]
당신은 수면 무호흡증(OSA) 및 수면 내시경(DISE) 해석에 특화된 이비인후과 전문의입니다.

[분석 데이터]
{context}

[답변 지침]
1. 반드시 한국어로 답변하세요.
2. 위 데이터에 근거해서만 답변하고, 데이터가 없으면 "데이터 부족"을 분명히 언급하세요.
3. 가능하면 수치(시간, 감소율, 이벤트 개수)를 인용하여 구체적으로 설명하세요.
4. 임상적 의미(경증/중등도/중증, 추적 필요 여부, 치료 권고)를 간단히 덧붙이세요.
5. 너무 장황하지 않게, 3~6문장 정도로 요약해서 답변하세요.
6. 이전 대화 맥락을 고려하여 자연스럽게 답변하세요.
"""

            # 대화 히스토리가 있으면 채팅 세션 사용
            if conversation_history and len(conversation_history) > 0:
                # Gemini 채팅 히스토리 형식으로 변환
                history = []
                for msg in conversation_history:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'user':
                        history.append({"role": "user", "parts": [content]})
                    elif role == 'assistant':
                        history.append({"role": "model", "parts": [content]})
                
                # 채팅 세션 시작
                model = genai.GenerativeModel(
                    "gemini-2.0-flash-exp", 
                    safety_settings=safety_settings,
                    system_instruction=system_prompt
                )
                chat = model.start_chat(history=history)
                
                # 현재 질문 전송
                resp = chat.send_message(question)
            else:
                # 첫 대화: 시스템 프롬프트와 질문을 함께 전송
                model = genai.GenerativeModel(
                    "gemini-2.0-flash-exp", 
                    safety_settings=safety_settings,
                    system_instruction=system_prompt
                )
                resp = model.generate_content(question)
            
            return {"success": True, "answer": resp.text}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"AI 답변 생성 실패: {str(e)}"}
    # ============================================================

    def generate_ai_summary(self):
        if not self.api_key: return "API Key Not Found."
        try:
            genai.configure(api_key=self.api_key)
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            
            model = genai.GenerativeModel('gemini-2.0-flash-exp', safety_settings=safety_settings)
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
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
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
        
        # --- [수정] Reference 이미지 표시 로직 (Manual 우선) ---
        ref_images_html = ""
        
        # 1. 수동 업로드 이미지가 존재하는 경우 (최우선 표시)
        if self.results.get('manual_ref_image'):
            # 파일명만 추출하여 웹 경로(overlays 폴더)로 변환
            manual_path = Path(self.results['manual_ref_image'])
            web_path = f"overlays/{manual_path.name}"
            
            ref_images_html = f"""
            <div class="space-y-3">
                <div class="relative group">
                    <div class="text-xs font-bold mb-1 text-indigo-600">📸 Manual Reference Used</div>
                    <img src="{web_path}" class="w-full rounded-lg border-4 border-indigo-500 shadow-md">
                    <div class="text-xs text-right mt-1 text-slate-400">
                        Max Area: {self.results.get('max_area', 0):.0f} px²
                    </div>
                </div>
            </div>
            """
            
        # 2. 수동 이미지가 없고, Auto 이미지가 있는 경우 (기존 로직)
        elif self.reference_images:
            ref_images_html = "<div class='space-y-3'>"
            for label in ['OTE', 'Velum']:
                if label in self.reference_images:
                    img_path = Path(self.reference_images[label])
                    web_path = f"overlays/{img_path.name}"
                    ref_data = self.segment_references.get(label, {})
                    color = '#3498db' if label == 'OTE' else '#9b59b6'
                    
                    ref_images_html += f"""
                    <div class="relative group">
                        <div class="text-xs font-bold mb-1" style="color: {color}">{label} Reference (Auto)</div>
                        <img src="{web_path}" class="w-full rounded-lg border-4 shadow-md transition-transform group-hover:scale-[1.02]" style="border-color: {color}">
                        <div class="text-xs text-right mt-1 text-slate-400">Max: {ref_data.get('max_area', 0):.0f} px²</div>
                    </div>
                    """
            ref_images_html += "</div>"
        
        # 3. 아무것도 없는 경우
        else:
            ref_images_html = "<div class='bg-gray-100 p-4 rounded text-center text-gray-500'>No Reference Images</div>"
        # -----------------------------------------------------------

        p_info = self.patient_info
        # ✅ 핵심 수정: video_stem을 실제 값으로 설정
        video_filename = self.video_info.get('filename', '')
        video_stem = Path(video_filename).stem if video_filename else ''

        # ========== VQA 섹션 HTML (채팅 인터페이스) ==========
        vqa_section = """
        <div class="card border-t-4 border-t-emerald-500 overflow-hidden">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-bold text-emerald-700 flex items-center gap-2">
                    <i class="fas fa-comments"></i> AI 대화형 질의응답
                </h3>
                <button onclick="clearChat()" class="text-xs px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg transition flex items-center gap-1">
                    <i class="fas fa-trash-alt"></i> 대화 초기화
                </button>
            </div>
            
            <!-- 채팅 영역 -->
            <div id="chatContainer" class="bg-slate-50 rounded-lg border border-slate-200 mb-4" style="height: 500px; overflow-y: auto;">
                <div id="chatMessages" class="p-4 space-y-4">
                    <!-- 환영 메시지 -->
                    <div class="flex items-start gap-3">
                        <div class="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center flex-shrink-0">
                            <i class="fas fa-robot text-emerald-600 text-sm"></i>
                        </div>
                        <div class="flex-1 bg-white rounded-lg p-3 shadow-sm border border-slate-200">
                            <p class="text-sm text-slate-700">
                                안녕하세요! 분석 결과에 대해 궁금한 점을 물어보세요. 
                                <span class="text-emerald-600 font-medium">대화를 이어가며</span> 더 자세한 정보를 얻을 수 있습니다.
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 입력 영역 -->
            <div class="space-y-3">
                <div class="flex gap-2">
                    <input type="text" id="vqaQuestion"
                           class="flex-1 px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                           placeholder="질문을 입력하세요... (Enter로 전송)"
                           onkeypress="if(event.key === 'Enter') askAI()">
                    <button onclick="askAI()" id="sendButton"
                            class="px-6 py-3 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 transition flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed">
                        <i class="fas fa-paper-plane"></i> 전송
                    </button>
                </div>

                <!-- 빠른 질문 버튼 -->
                <div class="flex flex-wrap gap-2">
                    <button onclick="setQuestion('가장 심각한 폐색 이벤트는 언제 발생했나요?')" 
                            class="text-xs px-3 py-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 rounded-full transition border border-emerald-200">
                        가장 심각한 이벤트는?
                    </button>
                    <button onclick="setQuestion('OTE와 Velum 중 어느 부위에서 폐색이 더 많이 발생했나요?')" 
                            class="text-xs px-3 py-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 rounded-full transition border border-emerald-200">
                        어느 부위가 더 심각한가요?
                    </button>
                    <button onclick="setQuestion('전체 폐색 이벤트의 평균 지속 시간은 얼마나 되나요?')" 
                            class="text-xs px-3 py-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 rounded-full transition border border-emerald-200">
                        평균 지속 시간은?
                    </button>
                    <button onclick="setQuestion('치료 권고사항이 있나요?')" 
                            class="text-xs px-3 py-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 rounded-full transition border border-emerald-200">
                        치료 권고사항
                    </button>
                </div>
            </div>
        </div>
        """

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
        <div class="flex items-center gap-3 font-bold text-xl"><i class="fas fa-heartbeat text-rose-500"></i> DISE AI Analytics</div>
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
                    <h3 class="text-sm font-bold text-slate-400 uppercase mb-2">Reference Image</h3>
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

        {vqa_section}

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
                        <th class="px-6 py-3">Max Reduction</th>
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
                reduction_val = event.get('max_reduction', 0)
                
                html += f"""
                    <tr onclick="playVideo('{video_path}', 'Event #{i+1}')" class="hover:bg-slate-50 cursor-pointer transition">
                        <td class="px-6 py-4"><span class="severity-badge s-{event['severity']}">{event['severity']}</span></td>
                        <td class="px-6 py-4 font-bold text-slate-700">{event['segment_label']}</td>
                        <td class="px-6 py-4 text-slate-500">{event['start_time']:.1f}s ~ {event['end_time']:.1f}s</td>
                        <td class="px-6 py-4 font-bold text-red-600">{reduction_val:.1f}%</td>
                        <td class="px-6 py-4 text-slate-600">{ref_area:.0f} px²</td>
                        <td class="px-6 py-4 text-center">
                            <button class="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 hover:bg-indigo-600 hover:text-white transition">
                                <i class="fas fa-play text-xs"></i>
                            </button>
                        </td>
                    </tr>
                """

        html += f"""
                </tbody>
            </table>
        </div>
    </div>

    <div id="videoModal" class="fixed inset-0 z-[100] hidden" onclick="closeModal()">
        <div class="fixed inset-0 bg-slate-900/90 backdrop-blur-sm"></div>
        <div class="fixed inset-0 flex items-center justify-center p-4">
            <div class="bg-black rounded-2xl shadow-2xl overflow-hidden max-w-7xl w-full relative" onclick="event.stopPropagation()">
                <div class="bg-slate-800 px-4 py-3 flex justify-between items-center">
                    <h3 class="text-white font-bold" id="modalTitle">Event Video</h3>
                    <button onclick="closeModal()" class="text-slate-400 hover:text-white"><i class="fas fa-times text-xl"></i></button>
                </div>
                <div class="bg-black flex items-center justify-center" style="min-height: 400px;">
                    <video id="player" controls class="w-full h-auto max-h-[80vh]" style="object-fit: contain;"></video>
                </div>
            </div>
        </div>
    </div>

    <script>
        // ✅ 핵심 수정: Python에서 실제 값을 주입
        const currentVideoStem = "{video_stem}";
        console.log("Current video_stem:", currentVideoStem);

        // ===== Multi-turn VQA 채팅 인터페이스 =====
        let conversationHistory = [];

        function setQuestion(text) {{
            const input = document.getElementById('vqaQuestion');
            if (input) {{
                input.value = text.trim();
                input.focus();
            }}
        }}

        function addMessage(role, content) {{
            const chatMessages = document.getElementById('chatMessages');
            if (!chatMessages) return;

            const messageDiv = document.createElement('div');
            messageDiv.className = 'flex items-start gap-3 animate-fadeIn';
            
            if (role === 'user') {{
                messageDiv.innerHTML = `
                    <div class="flex-1"></div>
                    <div class="flex items-start gap-3 flex-row-reverse max-w-[80%]">
                        <div class="w-8 h-8 rounded-full bg-emerald-500 flex items-center justify-center flex-shrink-0">
                            <i class="fas fa-user text-white text-xs"></i>
                        </div>
                        <div class="bg-emerald-500 text-white rounded-lg p-3 shadow-sm">
                            <p class="text-sm whitespace-pre-wrap">${{content}}</p>
                        </div>
                    </div>
                `;
            }} else {{
                messageDiv.innerHTML = `
                    <div class="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center flex-shrink-0">
                        <i class="fas fa-robot text-emerald-600 text-sm"></i>
                    </div>
                    <div class="flex-1 bg-white rounded-lg p-3 shadow-sm border border-slate-200 max-w-[80%]">
                        <p class="text-sm text-slate-700 whitespace-pre-wrap">${{content}}</p>
                    </div>
                `;
            }}
            
            chatMessages.appendChild(messageDiv);
            scrollToBottom();
        }}

        function addLoadingMessage() {{
            const chatMessages = document.getElementById('chatMessages');
            if (!chatMessages) return;

            const loadingDiv = document.createElement('div');
            loadingDiv.id = 'loadingMessage';
            loadingDiv.className = 'flex items-start gap-3';
            loadingDiv.innerHTML = `
                <div class="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-robot text-emerald-600 text-sm"></i>
                </div>
                <div class="flex-1 bg-white rounded-lg p-3 shadow-sm border border-slate-200">
                    <div class="flex items-center gap-2">
                        <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-emerald-600"></div>
                        <p class="text-sm text-slate-500">AI가 답변을 생성하고 있습니다...</p>
                    </div>
                </div>
            `;
            chatMessages.appendChild(loadingDiv);
            scrollToBottom();
        }}

        function removeLoadingMessage() {{
            const loadingMsg = document.getElementById('loadingMessage');
            if (loadingMsg) {{
                loadingMsg.remove();
            }}
        }}

        function scrollToBottom() {{
            const container = document.getElementById('chatContainer');
            if (container) {{
                container.scrollTop = container.scrollHeight;
            }}
        }}

        async function askAI() {{
            const input = document.getElementById('vqaQuestion');
            const sendButton = document.getElementById('sendButton');
            
            if (!input) return;
            const question = input.value.trim();
            if (!question) {{
                alert('질문을 입력해주세요.');
                return;
            }}

            // 입력 비활성화
            input.disabled = true;
            sendButton.disabled = true;

            // 사용자 메시지 추가
            addMessage('user', question);
            conversationHistory.push({{'role': 'user', 'content': question}});

            // 입력창 초기화
            input.value = '';

            // 로딩 메시지 추가
            addLoadingMessage();

            try {{
                console.log("Sending VQA request:", {{question, video_stem: currentVideoStem, history_length: conversationHistory.length}});
                
                const res = await fetch('/api/vqa', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        question: question,
                        video_stem: currentVideoStem,
                        conversation_history: conversationHistory.slice(0, -1)  // 현재 질문 제외한 히스토리
                    }})
                }});

                const data = await res.json();
                console.log("VQA response:", data);
                
                removeLoadingMessage();

                if (data.success) {{
                    const answer = data.answer || '';
                    addMessage('assistant', answer);
                    conversationHistory.push({{'role': 'assistant', 'content': answer}});
                }} else {{
                    addMessage('assistant', '죄송합니다. 오류가 발생했습니다: ' + (data.error || '알 수 없는 오류'));
                }}
            }} catch (err) {{
                removeLoadingMessage();
                console.error(err);
                addMessage('assistant', '서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
            }} finally {{
                // 입력 활성화
                input.disabled = false;
                sendButton.disabled = false;
                input.focus();
            }}
        }}

        function clearChat() {{
            if (!confirm('대화 기록을 모두 삭제하시겠습니까?')) return;
            
            conversationHistory = [];
            const chatMessages = document.getElementById('chatMessages');
            if (chatMessages) {{
                chatMessages.innerHTML = `
                    <div class="flex items-start gap-3">
                        <div class="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center flex-shrink-0">
                            <i class="fas fa-robot text-emerald-600 text-sm"></i>
                        </div>
                        <div class="flex-1 bg-white rounded-lg p-3 shadow-sm border border-slate-200">
                            <p class="text-sm text-slate-700">
                                대화가 초기화되었습니다. 새로운 질문을 해주세요.
                            </p>
                        </div>
                    </div>
                `;
            }}
        }}

        // Enter 키 이벤트
        const vqaInputEl = document.getElementById('vqaQuestion');
        if (vqaInputEl) {{
            vqaInputEl.addEventListener('keypress', (e) => {{
                if (e.key === 'Enter' && !e.shiftKey) {{
                    e.preventDefault();
                    if (!vqaInputEl.disabled) {{
                        askAI();
                    }}
                }}
            }});
        }}

        // 스타일 추가
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .animate-fadeIn {{
                animation: fadeIn 0.3s ease-out;
            }}
            #chatContainer::-webkit-scrollbar {{
                width: 8px;
            }}
            #chatContainer::-webkit-scrollbar-track {{
                background: #f1f5f9;
                border-radius: 4px;
            }}
            #chatContainer::-webkit-scrollbar-thumb {{
                background: #cbd5e1;
                border-radius: 4px;
            }}
            #chatContainer::-webkit-scrollbar-thumb:hover {{
                background: #94a3b8;
            }}
        `;
        document.head.appendChild(style);
        // ===================

        function playVideo(src, title) {{
            const player = document.getElementById('player');
            player.innerHTML = '';
            player.onerror = null;
            
            const source = document.createElement('source');
            source.src = src;
            source.type = 'video/mp4';
            player.appendChild(source);
            
            player.onerror = function(e) {{
                alert('비디오 파일을 재생할 수 없습니다.\\n경로: ' + src);
            }};
            
            document.getElementById('modalTitle').innerText = title + ' (좌측: 원본, 우측: 분석 결과)';
            document.getElementById('videoModal').classList.remove('hidden');
            
            player.load();
            player.play().catch(console.log);
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