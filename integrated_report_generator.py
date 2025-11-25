"""
통합 DISE 분석 보고서 생성기
- 구간별 분석
- 이벤트별 비디오 클립 표시
"""

from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


class IntegratedReportGenerator:
    """통합 보고서 생성기"""
    
    def __init__(self, results):
        """
        Args:
            results: IntegratedDISEAnalyzer의 분석 결과
        """
        self.results = results
        self.video_info = results['video_info']
        self.segments = results['segments']
        self.events = results['occlusion_events']
        self.summary = results['summary']
    
    def generate_timeline_chart(self, output_dir):
        """타임라인 차트 생성"""
        output_dir = Path(output_dir)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 배경색
        ax.set_facecolor('#f8f9fa')
        
        # 구간 표시
        colors = {'OTE': '#3498db', 'Velum': '#9b59b6', 'None': '#95a5a6'}
        
        for segment in self.segments:
            ax.barh(
                1, 
                segment['duration'], 
                left=segment['start_time'],
                height=0.3,
                color=colors[segment['label']],
                alpha=0.6,
                label=segment['label'] if segment == self.segments[0] or 
                      segment['label'] != self.segments[self.segments.index(segment)-1]['label'] 
                      else ""
            )
        
        # 폐색 이벤트 표시
        severity_colors = {
            'Mild': '#f39c12',
            'Moderate': '#e67e22',
            'Severe': '#e74c3c',
            'Critical': '#c0392b'
        }
        
        for event in self.events:
            ax.barh(
                0.5,
                event['duration'],
                left=event['start_time'],
                height=0.2,
                color=severity_colors[event['severity']],
                alpha=0.8
            )
            
            # 이벤트 번호 표시
            event_num = self.events.index(event) + 1
            ax.text(
                event['start_time'] + event['duration']/2,
                0.5,
                f"#{event_num}",
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color='white'
            )
        
        ax.set_xlabel('시간 (초)', fontsize=12)
        ax.set_xlim(0, self.video_info['duration'])
        ax.set_ylim(0, 1.5)
        ax.set_yticks([0.5, 1])
        ax.set_yticklabels(['폐색 이벤트', '구간 분류'])
        ax.set_title('비디오 타임라인 분석', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 범례
        handles = [
            plt.Rectangle((0,0),1,1, color=colors['OTE'], alpha=0.6, label='OTE'),
            plt.Rectangle((0,0),1,1, color=colors['Velum'], alpha=0.6, label='Velum'),
            plt.Rectangle((0,0),1,1, color=severity_colors['Mild'], alpha=0.8, label='Mild'),
            plt.Rectangle((0,0),1,1, color=severity_colors['Moderate'], alpha=0.8, label='Moderate'),
            plt.Rectangle((0,0),1,1, color=severity_colors['Severe'], alpha=0.8, label='Severe'),
            plt.Rectangle((0,0),1,1, color=severity_colors['Critical'], alpha=0.8, label='Critical'),
        ]
        ax.legend(handles=handles, loc='upper right', ncol=2)
        
        plt.tight_layout()
        
        chart_path = output_dir / 'timeline.png'
        plt.savefig(str(chart_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def generate_severity_chart(self, output_dir):
        """심각도별 통계 차트"""
        output_dir = Path(output_dir)
        
        severities = ['Mild', 'Moderate', 'Severe', 'Critical']
        counts = [self.summary['events_by_severity'][s] for s in severities]
        colors = ['#f39c12', '#e67e22', '#e74c3c', '#c0392b']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 막대 차트
        bars = ax1.bar(severities, counts, color=colors, alpha=0.7)
        ax1.set_xlabel('심각도', fontsize=12)
        ax1.set_ylabel('이벤트 수', fontsize=12)
        ax1.set_title('심각도별 이벤트 수', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 파이 차트
        non_zero = [(s, c) for s, c in zip(severities, counts) if c > 0]
        if non_zero:
            labels, values = zip(*non_zero)
            pie_colors = [colors[severities.index(s)] for s in labels]
            ax2.pie(values, labels=labels, colors=pie_colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            ax2.set_title('심각도 분포', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        chart_path = output_dir / 'severity_chart.png'
        plt.savefig(str(chart_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def generate_html_report(self, output_dir):
        """HTML 보고서 생성"""
        output_dir = Path(output_dir)
        
        # 차트 생성
        timeline_chart = self.generate_timeline_chart(output_dir)
        severity_chart = self.generate_severity_chart(output_dir)
        
        # HTML 생성
        html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DISE 통합 분석 보고서</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 16px;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 24px;
            color: #2c3e50;
            border-left: 4px solid #667eea;
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .info-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .info-card .label {{
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        
        .info-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .chart {{
            text-align: center;
            margin: 30px 0;
        }}
        
        .chart img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .event-card {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s;
        }}
        
        .event-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        
        .event-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .event-number {{
            font-size: 18px;
            font-weight: bold;
            color: #667eea;
        }}
        
        .severity-badge {{
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        
        .severity-mild {{ background: #f39c12; color: white; }}
        .severity-moderate {{ background: #e67e22; color: white; }}
        .severity-severe {{ background: #e74c3c; color: white; }}
        .severity-critical {{ background: #c0392b; color: white; }}
        
        .event-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .detail-item {{
            font-size: 14px;
        }}
        
        .detail-label {{
            color: #6c757d;
            font-size: 12px;
        }}
        
        .detail-value {{
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .video-player {{
            margin-top: 15px;
            text-align: center;
        }}
        
        .video-player video {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        
        .no-events {{
            text-align: center;
            padding: 40px;
            color: #6c757d;
            font-size: 16px;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }}
        
        .segment-list {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .segment-item {{
            padding: 10px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .segment-item:last-child {{
            border-bottom: none;
        }}
        
        .segment-label {{
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }}
        
        .segment-ote {{
            background: #3498db;
            color: white;
        }}
        
        .segment-velum {{
            background: #9b59b6;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 DISE 통합 분석 보고서</h1>
            <p>수면 내시경 기도 폐색 자동 분석 시스템</p>
        </div>
        
        <div class="content">
            <!-- 비디오 정보 -->
            <div class="section">
                <h2 class="section-title">📹 비디오 정보</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <div class="label">파일명</div>
                        <div class="value" style="font-size: 16px;">{self.video_info['filename']}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">길이</div>
                        <div class="value">{self.video_info['duration']:.1f}초</div>
                    </div>
                    <div class="info-card">
                        <div class="label">FPS</div>
                        <div class="value">{self.video_info['fps']:.1f}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">총 프레임</div>
                        <div class="value">{self.video_info['total_frames']}</div>
                    </div>
                </div>
            </div>
            
            <!-- 분석 요약 -->
            <div class="section">
                <h2 class="section-title">📊 분석 요약</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <div class="label">총 구간</div>
                        <div class="value">{self.summary['total_segments']}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">OTE 구간</div>
                        <div class="value">{self.summary['ote_segments']}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Velum 구간</div>
                        <div class="value">{self.summary['velum_segments']}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">폐색 이벤트</div>
                        <div class="value">{self.summary['total_events']}</div>
                    </div>
                </div>
            </div>
            
            <!-- 타임라인 차트 -->
            <div class="section">
                <h2 class="section-title">📈 타임라인 분석</h2>
                <div class="chart">
                    <img src="timeline.png" alt="Timeline Chart">
                </div>
            </div>
            
            <!-- 심각도 차트 -->
            <div class="section">
                <h2 class="section-title">⚠️ 심각도 분석</h2>
                <div class="chart">
                    <img src="severity_chart.png" alt="Severity Chart">
                </div>
            </div>
            
            <!-- 구간 목록 -->
            <div class="section">
                <h2 class="section-title">🎬 구간 목록</h2>
                <div class="segment-list">
        """
        
        for i, segment in enumerate(self.segments, 1):
            label_class = f"segment-{segment['label'].lower()}"
            html += f"""
                    <div class="segment-item">
                        <div>
                            <span class="segment-label {label_class}">{segment['label']}</span>
                            <span style="margin-left: 10px;">구간 #{i}</span>
                        </div>
                        <div class="detail-value">
                            {segment['start_time']:.1f}s ~ {segment['end_time']:.1f}s 
                            ({segment['duration']:.1f}s)
                        </div>
                    </div>
            """
        
        html += """
                </div>
            </div>
            
            <!-- 폐색 이벤트 -->
            <div class="section">
                <h2 class="section-title">🚨 폐색 이벤트 상세</h2>
        """
        
        if self.events:
            for i, event in enumerate(self.events, 1):
                severity_class = f"severity-{event['severity'].lower()}"
                clip_path = event.get('clip_path', '')
                
                html += f"""
                <div class="event-card">
                    <div class="event-header">
                        <span class="event-number">이벤트 #{i}</span>
                        <span class="severity-badge {severity_class}">{event['severity']}</span>
                    </div>
                    <div class="event-details">
                        <div class="detail-item">
                            <div class="detail-label">구간</div>
                            <div class="detail-value">{event['segment_label']}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">시작</div>
                            <div class="detail-value">{event['start_time']:.1f}초</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">종료</div>
                            <div class="detail-value">{event['end_time']:.1f}초</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">지속 시간</div>
                            <div class="detail-value">{event['duration']:.1f}초</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">최대 감소율</div>
                            <div class="detail-value">{event['max_reduction']:.1f}%</div>
                        </div>
                    </div>
                """
                
                if clip_path and Path(clip_path).exists():
                    rel_path = Path(clip_path).name
                    html += f"""
                    <div class="video-player">
                        <video controls width="640">
                            <source src="event_clips/{rel_path}" type="video/mp4">
                        </video>
                    </div>
                    """
                
                html += """
                </div>
                """
        else:
            html += """
                <div class="no-events">
                    ✅ 폐색 이벤트가 감지되지 않았습니다.
                </div>
            """
        
        html += f"""
            </div>
        </div>
        
        <div class="footer">
            <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>DISE Integrated Analysis System v2.0</p>
        </div>
    </div>
</body>
</html>
        """
        
        report_path = output_dir / 'report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return report_path
    
    def generate_report(self, output_dir):
        """전체 보고서 생성"""
        print("\n📝 보고서 생성 중...")
        report_path = self.generate_html_report(output_dir)
        print(f"✅ 보고서 생성 완료: {report_path}")
        return report_path


if __name__ == '__main__':
    # 테스트용
    with open('analysis_results.json') as f:
        results = json.load(f)
    
    generator = IntegratedReportGenerator(results)
    generator.generate_report('output')
