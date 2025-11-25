"""
수면 내시경 분석 웹 애플리케이션 (Flask 백엔드)
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import sys

# 모듈 import
sys.path.append(str(Path(__file__).parent))
from video_analyzer import AirwayOcclusionAnalyzer
from report_generator import ReportGenerator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 제한
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['OUTPUT_FOLDER'] = Path('outputs')

# 폴더 생성
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """비디오 분석 API"""
    try:
        # 파일 체크
        if 'video' not in request.files:
            return jsonify({'error': '비디오 파일이 없습니다.'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
        
        # 파라미터
        fps_extract = int(request.form.get('fps_extract', 5))
        threshold_percent = float(request.form.get('threshold_percent', 30))
        
        # 파일 저장
        filename = secure_filename(file.filename)
        video_path = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(video_path))
        
        # 분석 실행
        analyzer = AirwayOcclusionAnalyzer(
            fps_extract=fps_extract,
            threshold_percent=threshold_percent
        )
        
        output_dir = app.config['OUTPUT_FOLDER'] / Path(filename).stem
        results = analyzer.analyze_video(str(video_path), output_dir=str(output_dir))
        
        # 보고서 생성
        report_generator = ReportGenerator(results)
        html_report_path = report_generator.generate_full_report(output_dir)
        
        return jsonify({
            'success': True,
            'message': '분석이 완료되었습니다.',
            'report_url': f'/outputs/{Path(filename).stem}/report.html',
            'results': {
                'max_area': int(results['max_area']),
                'max_area_frame': int(results['max_area_frame']),
                'total_events': int(len(results['occlusion_events'])),
                'metadata': results['metadata']
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """출력 파일 제공"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/api/health')
def health_check():
    """헬스 체크"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 수면 내시경 분석 시스템 시작")
    print("=" * 60)
    print(f"  📂 업로드 폴더: {app.config['UPLOAD_FOLDER'].absolute()}")
    print(f"  📂 출력 폴더: {app.config['OUTPUT_FOLDER'].absolute()}")
    print(f"  🌐 주소: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
