"""
분석 결과 보고서 생성
- HTML 보고서
- 시각화 차트
"""

import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """NumPy 타입을 JSON 직렬화 가능하도록 변환"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ReportGenerator:
    """보고서 생성기"""
    
    def __init__(self, results_data):
        """
        Args:
            results_data: analyze_video()의 결과 딕셔너리
        """
        self.results = results_data
        self.metadata = results_data.get('metadata', {})
    
    def generate_reference_image(self, output_path):
        """기준 이미지 (최대 면적 프레임) 생성"""
        frames = self.results.get('frames', [])
        max_area_frame_idx = self.results.get('max_area_frame_index', 0)
        
        # 기준 프레임 찾기
        ref_frame = None
        for frame in frames:
            if frame.get('extracted_index') == max_area_frame_idx:
                ref_frame = frame
                break
        
        if ref_frame and 'overlay_path' in ref_frame:
            import shutil
            # overlays 폴더로 복사
            shutil.copy(ref_frame['overlay_path'], output_path)
            print(f"📸 기준 이미지 생성: {output_path}")
            return True
        
        return False
    
    def generate_area_chart(self, output_path):
        """ROI 면적 변화 그래프 생성"""
        frames = self.results.get('frames', [])
        
        timestamps = [f['timestamp'] for f in frames if f.get('roi_area', 0) > 0]
        areas = [f['roi_area'] for f in frames if f.get('roi_area', 0) > 0]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 면적 그래프
        ax.plot(timestamps, areas, 'b-', linewidth=2, label='ROI Area')
        ax.fill_between(timestamps, areas, alpha=0.3)
        
        # 최대 면적 기준선
        max_area = self.results.get('max_area', 0)
        if max_area > 0:
            ax.axhline(y=max_area, color='g', linestyle='--', linewidth=2, 
                      label=f'Max Area ({max_area:.0f} px²)')
            
            # 폐색 기준선
            threshold_percent = self.metadata.get('threshold_percent', 30)
            threshold_area = max_area * (1 - threshold_percent / 100)
            ax.axhline(y=threshold_area, color='r', linestyle='--', linewidth=2,
                      label=f'Occlusion Threshold ({threshold_percent}%)')
        
        # 폐색 이벤트 표시
        occlusion_events = self.results.get('occlusion_events', [])
        for event in occlusion_events:
            ax.axvline(x=event['timestamp'], color='orange', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('ROI Area (px²)', fontsize=12, fontweight='bold')
        ax.set_title('Airway Opening Area Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 면적 차트 생성: {output_path}")
    
    def generate_event_timeline(self, output_path):
        """폐색 이벤트 타임라인 시각화"""
        events = self.results.get('occlusion_events', [])
        
        if not events:
            print("⚠️  폐색 이벤트가 없어 타임라인을 생성하지 않습니다.")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        timestamps = [e['timestamp'] for e in events]
        reductions = [e['area_reduction_percent'] for e in events]
        severities = [e['severity'] for e in events]
        
        # 심각도별 색상
        severity_colors = {
            'Mild': 'yellow',
            'Moderate': 'orange',
            'Severe': 'red',
            'Critical': 'darkred'
        }
        
        colors = [severity_colors.get(s, 'gray') for s in severities]
        
        ax.scatter(timestamps, reductions, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Area Reduction (%)', fontsize=12, fontweight='bold')
        ax.set_title('Occlusion Events Timeline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 범례
        for severity, color in severity_colors.items():
            ax.scatter([], [], c=color, s=100, alpha=0.7, edgecolors='black', label=severity)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 이벤트 타임라인 생성: {output_path}")
    
    def generate_html_report(self, output_path, chart_path, timeline_path, reference_path=None):
        """
        HTML 보고서 생성 (심각도 기준 표 + 행 색상 포함)
        - f-string 대신 토큰 치환 방식이라 안전합니다.
        """
        import json
        from pathlib import Path
        from datetime import datetime

        events = self.results.get('occlusion_events', [])
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chart_name = Path(chart_path).name if chart_path else ""
        timeline_name = Path(timeline_path).name if timeline_path else ""

        # ── 이벤트 테이블
        if events:
            rows = []
            for i, e in enumerate(events):
                sev = e.get('severity', '')
                sev_cls = str(sev).lower()
                area = float(e.get('roi_area', 0))
                red = float(e.get('area_reduction_percent', 0.0))
                fno = int(e.get('frame_number', 0))
                tstr = e.get('time_str', '')
                has_img = bool(e.get('overlay_path')) and bool(e.get('frame_path'))
                cell = '🖼️ 클릭하여 보기' if has_img else '-'
                rows.append(
                    "<tr class=\"severity-{cls}\" onclick=\"showEventImages({idx})\" style=\"cursor:pointer;\">"
                    "<td>{t}</td><td>{f}</td><td>{a:.0f}</td><td>{r:.1f}%</td><td><strong>{s}</strong></td><td>{c}</td>"
                    "</tr>".format(cls=sev_cls, idx=i, t=tstr, f=fno, a=area, r=red, s=sev, c=cell)
                )
            table_block = (
                "<table id=\"eventTable\">"
                "<thead><tr>"
                "<th>시간</th><th>프레임 번호</th><th>ROI 면적 (px²)</th><th>감소율 (%)</th><th>심각도</th><th>이미지</th>"
                "</tr></thead><tbody>{rows}</tbody></table>"
            ).format(rows="".join(rows))
        else:
            table_block = "<p>폐색 이벤트가 감지되지 않았습니다.</p>"

        # ── 모달용 데이터(JSON)
        events_for_js = []
        for e in events:
            events_for_js.append({
                "overlay_path": e.get("overlay_path", ""),
                "frame_path": e.get("frame_path", ""),
                "time_str": e.get("time_str", ""),
                "severity": e.get("severity", ""),
                "reduction": "{:.1f}%".format(float(e.get("area_reduction_percent", 0.0))),
                "frame_number": int(e.get("frame_number", 0)),
            })
        try:
            events_json = json.dumps(events_for_js, ensure_ascii=False, cls=NumpyEncoder)  # noqa: F821
        except NameError:
            def _np_default(o):
                try:
                    import numpy as np
                    if isinstance(o, np.integer): return int(o)
                    if isinstance(o, np.floating): return float(o)
                    if isinstance(o, np.ndarray): return o.tolist()
                except Exception:
                    pass
                return str(o)
            events_json = json.dumps(events_for_js, ensure_ascii=False, default=_np_default)

        # ── 기준 이미지 블록(옵션)
        max_area = float(self.results.get('max_area', 0))
        max_area_frame = int(self.results.get('max_area_frame', 0))
        ref_block = ""
        if reference_path and Path(reference_path).exists():
            ref_filename = Path(reference_path).name
            ref_block = (
                "<div class=\"reference-section\">"
                "<h3>📸 기준 이미지 (최대 기도 개방 상태)</h3>"
                f"<p><strong>프레임 번호:</strong> {max_area_frame}</p>"
                f"<p><strong>ROI 면적:</strong> {max_area:.0f} px²</p>"
                "<p style=\"color:#666; margin-top:10px;\">"
                "💡 노란색 영역이 검출된 기도 개방 부분입니다. 이 면적을 기준으로 폐색 정도를 판단합니다."
                "</p>"
                f"<div style=\"text-align:center; margin-top:15px;\"><img src=\"overlays/{ref_filename}\" class=\"reference-image\" alt=\"기준 이미지\"></div>"
                "</div>"
            )

        # ── 차트/타임라인
        chart_img_html = f'<img src="{chart_name}" alt="Area Chart">' if chart_name else '<p>차트 없음</p>'
        timeline_img_html = (
            f'<div class="chart"><h2>⏱️ 폐색 이벤트 타임라인</h2><img src="{timeline_name}" alt="Timeline"></div>'
        ) if timeline_name else ''

        # ── 메타데이터
        video_file = str(self.metadata.get('video_file', 'N/A'))
        total_frames = int(self.metadata.get('total_frames', 0))
        duration_seconds = float(self.metadata.get('duration_seconds', 0.0))
        fps_val = float(self.metadata.get('fps', 0.0))
        extracted_frames = int(self.metadata.get('extracted_frames', 0))
        threshold_percent = self.metadata.get('threshold_percent', 30)
        generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # ── HTML 템플릿
        html_template = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1.0" />
    <title>수면 내시경 폐색 분석 보고서</title>
    <style>
    body { font-family: Arial, sans-serif; margin: 40px; background:#f5f5f5; }
    .container { max-width: 1400px; margin: 0 auto; background:#fff; padding: 40px; border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.1); }
    h1 { color:#2c3e50; border-bottom:3px solid #3498db; padding-bottom:10px; }
    h2 { color:#34495e; margin-top:30px; border-left:4px solid #3498db; padding-left:15px; }
    .metadata { background:#ecf0f1; padding:20px; border-radius:5px; margin:20px 0; }
    .metadata-item { margin:10px 0; font-size:16px; }
    .metadata-label { font-weight:bold; color:#2c3e50; display:inline-block; width:200px; }
    .chart { margin:30px 0; text-align:center; }
    .chart img { max-width:100%; border-radius:5px; box-shadow:0 2px 5px rgba(0,0,0,0.1); }
    .summary { background:#e3f2fd; padding:20px; border-radius:5px; margin:20px 0; border-left:5px solid #2196f3; }
    .summary-item { font-size:18px; margin:10px 0; }
    table { width:100%; border-collapse:collapse; margin:20px 0; }
    th, td { padding:12px; text-align:left; border-bottom:1px solid #ddd; }
    th { background:#3498db; color:#fff; font-weight:bold; }
    tr:hover { background:#f5f5f5; cursor:pointer; }
    /* 심각도 행 색상 */
    .severity-mild { background:#fff9c4; }        /* 연노랑 */
    .severity-moderate { background:#ffe0b2; }    /* 연주황 */
    .severity-severe { background:#ffcdd2; }      /* 연분홍 */
    .severity-critical { background:#ef5350; color:#fff; } /* 진빨강 */

    /* 심각도 기준 표 */
    .severity-guide { background:#fff3cd; padding:20px; border-radius:8px; margin:20px 0; border-left:5px solid #ffc107; }
    .severity-table { width:100%; border-collapse: collapse; margin-top:10px; }
    .severity-table th, .severity-table td { padding:12px; text-align:center; border:1px solid #ddd; }
    .severity-table th { background:#ffc107; color:#333; font-weight:bold; }

    .reference-section { background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%); padding:25px; border-radius:10px; margin:30px 0; border:3px solid #2196f3; }
    .reference-image { max-width:600px; border:3px solid #2196f3; border-radius:8px; box-shadow:0 4px 10px rgba(0,0,0,0.2); }

    .modal { display:none; position:fixed; z-index:1000; left:0; top:0; width:100%; height:100%; background:rgba(0,0,0,0.9); overflow:auto; }
    .modal-content { margin:2% auto; display:block; max-width:90%; max-height:90%; }
    .close { position:absolute; top:30px; right:50px; color:#f1f1f1; font-size:40px; font-weight:bold; cursor:pointer; }
    .close:hover { color:#ff4444; }
    .modal-caption { text-align:center; color:#ccc; padding:10px; font-size:20px; }
    .footer { margin-top:40px; padding-top:20px; border-top:2px solid #ddd; text-align:center; color:#7f8c8d; }
    </style>
    </head>
    <body>
    <div class="container">
    <h1>🔬 수면 내시경 기도 폐색 분석 보고서</h1>

    <div class="metadata">
        <h2>📋 비디오 정보</h2>
        <div class="metadata-item"><span class="metadata-label">파일명:</span><span>__VIDEO_FILE__</span></div>
        <div class="metadata-item"><span class="metadata-label">총 프레임 수:</span><span>__TOTAL_FRAMES__개</span></div>
        <div class="metadata-item"><span class="metadata-label">영상 길이:</span><span>__DURATION__초</span></div>
        <div class="metadata-item"><span class="metadata-label">FPS:</span><span>__FPS__</span></div>
        <div class="metadata-item"><span class="metadata-label">분석 프레임 수:</span><span>__EXTRACTED__개</span></div>
        <div class="metadata-item"><span class="metadata-label">폐색 기준:</span><span>기준 대비 __THRESHOLD__% 감소</span></div>
    </div>

    __REF_BLOCK__

    <div class="severity-guide">
        <h3>📊 폐색 심각도 분류 기준</h3>
        <table class="severity-table">
        <thead><tr><th>심각도</th><th>면적 감소율</th><th>임상적 의미</th><th>권장 조치</th></tr></thead>
        <tbody>
            <tr class="severity-mild"><td><strong>Mild (경미)</strong></td><td>30% - 50%</td><td>경미한 기도 협착</td><td>경과 관찰</td></tr>
            <tr class="severity-moderate"><td><strong>Moderate (중등도)</strong></td><td>50% - 70%</td><td>중등도 기도 폐색</td><td>치료 고려</td></tr>
            <tr class="severity-severe"><td><strong>Severe (심각)</strong></td><td>70% - 90%</td><td>심각한 기도 폐색</td><td>적극적 치료 권장</td></tr>
            <tr class="severity-critical"><td><strong>Critical (위중)</strong></td><td>90% 이상</td><td>거의 완전 폐색</td><td>즉시 치료 필요</td></tr>
        </tbody>
        </table>
    </div>

    <div class="summary">
        <h2>📊 분석 요약</h2>
        <div class="summary-item">✅ <strong>최대 기도 개방 면적:</strong> __MAX_AREA__ px² (프레임 __MAX_FRAME__)</div>
        <div class="summary-item">⚠️ <strong>폐색 이벤트 수:</strong> __EVENT_COUNT__개</div>
    </div>

    <div class="chart">
        <h2>📈 기도 개방 면적 변화</h2>
        __CHART__
    </div>

    __TIMELINE__

    <h2>🚨 폐색 이벤트 상세 (클릭하여 이미지 확인)</h2>
    <p style="color:#666; margin-bottom:20px;">💡 각 행을 클릭하면 해당 프레임의 상세 이미지를 확인할 수 있습니다.</p>

    __EVENT_TABLE__

    <!-- 이미지 모달 -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <img class="modal-content" id="modalImageOriginal" alt="원본 프레임">
        <img class="modal-content" id="modalImageOverlay" alt="Overlay" style="margin-top:10px;">
        <span class="close" onclick="closeModal()">&times;</span>
        <div class="modal-caption" id="modalCaption"></div>
    </div>

    <div class="footer">
        <p>생성 시간: __GENERATED__</p>
        <p>Airway Occlusion Analysis System v2.0</p>
        <p>의료진 판독 보조용 - 최종 진단은 전문의의 판단을 따르시기 바랍니다.</p>
    </div>
    </div>

    <script>
    const events = __EVENTS_JSON__;

    function showEventImages(index) {
    const e = events[index];
    if (!e) return;

    const toName = (p) => p ? p.split('/').pop().split('\\\\').pop() : '';
    const frameName = toName(e.frame_path);
    const overlayName = toName(e.overlay_path);

    const modal = document.getElementById('imageModal');
    const imgOrig = document.getElementById('modalImageOriginal');
    const imgOv = document.getElementById('modalImageOverlay');
    const cap = document.getElementById('modalCaption');

    imgOrig.src = 'frames/' + frameName;
    imgOv.src = 'overlays/' + overlayName;

    cap.innerHTML = `<strong>시간: ${e.time_str}</strong> | <strong>프레임: ${e.frame_number}</strong> | 감소율: <strong>${e.reduction}</strong>`;
    modal.style.display = 'block';
    }

    function closeModal() {
    document.getElementById('imageModal').style.display = 'none';
    }

    document.addEventListener('keydown', function(ev) {
    if (ev.key === 'Escape') closeModal();
    });
    </script>
    </body>
    </html>
    """

        html_content = (html_template
            .replace("__VIDEO_FILE__", video_file)
            .replace("__TOTAL_FRAMES__", "{:,}".format(total_frames))
            .replace("__DURATION__", "{:.2f}".format(duration_seconds))
            .replace("__FPS__", "{:.2f}".format(fps_val))
            .replace("__EXTRACTED__", str(extracted_frames))
            .replace("__THRESHOLD__", str(threshold_percent))
            .replace("__MAX_AREA__", "{:.0f}".format(max_area))
            .replace("__MAX_FRAME__", str(max_area_frame))
            .replace("__EVENT_COUNT__", str(len(events)))
            .replace("__CHART__", chart_img_html)
            .replace("__TIMELINE__", timeline_img_html)
            .replace("__EVENT_TABLE__", table_block)
            .replace("__EVENTS_JSON__", events_json)
            .replace("__GENERATED__", generated_time)
            .replace("__REF_BLOCK__", ref_block)
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"📄 HTML 보고서 생성: {output_path}")


        
    def generate_full_report(self, output_dir):
        """전체 보고서 생성"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 기준 이미지 생성
        reference_path = output_path / "overlays" / "reference_frame.jpg"
        self.generate_reference_image(reference_path)
        
        # 차트 생성
        chart_path = output_path / "area_chart.png"
        timeline_path = output_path / "event_timeline.png"
        
        self.generate_area_chart(chart_path)
        self.generate_event_timeline(timeline_path)
        
        # HTML 보고서
        html_path = output_path / "report.html"
        self.generate_html_report(
            html_path, 
            chart_path, 
            timeline_path,
            reference_path if reference_path.exists() else None
        )
        
        # JSON 결과 저장
        json_path = output_path / "results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"\n✅ 보고서 생성 완료!")
        print(f"  📂 출력 디렉토리: {output_path}")
        print(f"  📄 HTML 보고서: {html_path}")
        print(f"  📊 데이터: {json_path}")
        
        return html_path


# ========== 테스트 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("보고서 생성기 테스트")
    print("=" * 60)
    
    # 샘플 데이터로 테스트
    sample_results = {
        'max_area': 15000,
        'max_area_frame': 50,
        'frames': [
            {'timestamp': 0, 'roi_area': 12000},
            {'timestamp': 1, 'roi_area': 14000},
            {'timestamp': 2, 'roi_area': 15000},
            {'timestamp': 3, 'roi_area': 10000},
            {'timestamp': 4, 'roi_area': 8000},
        ],
        'occlusion_events': [
            {
                'frame_number': 75,
                'timestamp': 3.0,
                'time_str': '0:00:03',
                'roi_area': 10000,
                'area_reduction_percent': 33.3,
                'severity': 'Moderate'
            },
            {
                'frame_number': 100,
                'timestamp': 4.0,
                'time_str': '0:00:04',
                'roi_area': 8000,
                'area_reduction_percent': 46.7,
                'severity': 'Severe'
            }
        ],
        'metadata': {
            'video_file': 'test_video.mp4',
            'total_frames': 300,
            'extracted_frames': 60,
            'fps': 25.0,
            'duration_seconds': 12.0,
            'extraction_fps': 5,
            'threshold_percent': 30
        }
    }
    
    generator = ReportGenerator(sample_results)
    generator.generate_full_report('/home/claude/test_report')
