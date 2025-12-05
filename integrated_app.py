from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import sys
import json
import cv2
import numpy as np

# 통합 분석기 import
sys.path.append(str(Path(__file__).parent))
from integrated_analyzer import IntegratedDISEAnalyzer
from integrated_report_generator import IntegratedReportGenerator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['OUTPUT_FOLDER'] = Path('outputs')
app.config['MODEL_PATH'] = Path('ote_velum_classification_final/checkpoints/best_model.pth')
app.config['GEMINI_API_KEY'] = "AIzaSyCNtQzta2v9stW17EZtiT6ICKAIZawORY8" 

app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(exist_ok=True)


def calculate_manual_max_area(image_path, analyzer):
    """
    Reference 이미지에서 ROI를 분석하여 max_area 계산
    
    Args:
        image_path: Reference 이미지 경로
        analyzer: IntegratedDISEAnalyzer 인스턴스 (preprocess_frame 등 사용)
    
    Returns:
        max_area: 계산된 면적 (실패 시 None)
    """
    try:
        # 이미지 로드
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"⚠️ Failed to load reference image: {image_path}")
            return None
        
        # 전처리 (검은 배경 제거)
        preprocessed, bbox = analyzer.preprocess_frame(frame)
        
        # OTE와 Velum 둘 다 시도해서 더 큰 값 사용
        max_area_ote, _ = analyzer.analyze_roi_dual_track(preprocessed, 'OTE')
        max_area_velum, _ = analyzer.analyze_roi_dual_track(preprocessed, 'Velum')
        
        manual_max_area = max(max_area_ote, max_area_velum)
        
        if manual_max_area > 0:
            print(f"✅ Manual max_area calculated: {manual_max_area:.0f} px²")
            print(f"   (OTE: {max_area_ote:.0f}, Velum: {max_area_velum:.0f})")
            return manual_max_area
        else:
            print(f"⚠️ Failed to detect ROI in reference image")
            return None
            
    except Exception as e:
        print(f"❌ Error calculating manual max_area: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400
        
        file = request.files['video']
        filename = secure_filename(file.filename)
        video_path = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(video_path))
        
        # 사용자 입력 파라미터 가져오기
        fps_extract = int(request.form.get('fps_extract', 5))
        threshold_percent = int(request.form.get('threshold_percent', 30))
        min_event_duration = float(request.form.get('min_event_duration', 1.0))
        
        # 메타데이터 처리
        patient_info = {}
        if 'metadata' in request.files and request.files['metadata'].filename != '':
            try:
                meta = json.load(request.files['metadata'])
                if 'metas' in meta: patient_info = meta['metas']
                if 'videos' in meta: patient_info.update(meta['videos'])
            except: pass

        # ⭐ 기준 이미지 처리 및 manual_max_area 계산
        manual_max_area = None
        manual_ref_path = None
        
        if 'reference_image' in request.files:
            ref_file = request.files['reference_image']
            if ref_file.filename != '':
                ref_save_dir = app.config['OUTPUT_FOLDER'] / Path(filename).stem / 'overlays'
                ref_save_dir.mkdir(parents=True, exist_ok=True)
                manual_ref_path = ref_save_dir / 'manual_reference.jpg'
                ref_file.save(str(manual_ref_path))
                
                # 임시 분석기 생성 (manual_max_area 계산용)
                temp_analyzer = IntegratedDISEAnalyzer(
                    model_path=str(app.config['MODEL_PATH']),
                    fps_extract=fps_extract,
                    threshold_percent=threshold_percent,
                    min_event_duration=min_event_duration
                )
                
                # Reference 이미지에서 max_area 계산
                manual_max_area = calculate_manual_max_area(manual_ref_path, temp_analyzer)
                
                if manual_max_area:
                    print(f"📌 Using manual max_area: {manual_max_area:.0f} px²")
                else:
                    print(f"⚠️ Failed to calculate manual max_area, will use auto-detection")

        # 분석 실행 - manual_max_area 포함
        analyzer = IntegratedDISEAnalyzer(
            model_path=str(app.config['MODEL_PATH']),
            fps_extract=fps_extract,
            threshold_percent=threshold_percent,
            min_event_duration=min_event_duration,
            manual_max_area=manual_max_area  # ✅ 추가!
        )
        
        output_dir = app.config['OUTPUT_FOLDER'] / Path(filename).stem
        
        results = analyzer.analyze_video(str(video_path), output_dir=str(output_dir))
        
        results['patient_info'] = patient_info
        
        # 수동 기준 이미지 경로를 결과에 추가
        if manual_ref_path and manual_ref_path.exists():
            results['manual_ref_image'] = str(manual_ref_path)

        # 보고서 생성
        report_gen = IntegratedReportGenerator(results, api_key=app.config['GEMINI_API_KEY'])
        report_gen.generate_report(output_dir)
        
        return jsonify({
            'success': True,
            'report_url': f'/outputs/{Path(filename).stem}/report.html'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/vqa', methods=['POST', 'OPTIONS'])
def vqa():
    """
    VQA 엔드포인트: 분석 결과 기반 질의응답
    
    Request JSON:
        {
            "question": "...",
            "video_stem": "30042181_89_OTEclip"
        }
    """
    # CORS preflight 처리
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        print("=" * 50)
        print("🔍 VQA 요청 수신")
        
        data = request.get_json()
        print(f"📥 요청 데이터: {data}")
        
        question = (data.get('question') or '').strip()
        video_stem = (data.get('video_stem') or '').strip()

        print(f"❓ 질문: {question}")
        print(f"📁 video_stem: {video_stem}")

        if not question:
            return jsonify({'success': False, 'error': '질문을 입력해주세요.'}), 400
        if not video_stem:
            return jsonify({'success': False, 'error': 'video_stem이 필요합니다.'}), 400

        # 분석 결과 JSON 로드
        result_path = app.config['OUTPUT_FOLDER'] / video_stem / 'analysis_results.json'
        print(f"🔍 결과 파일 경로: {result_path}")
        print(f"✅ 파일 존재 여부: {result_path.exists()}")
        
        # outputs 폴더 내용 확인 (디버깅)
        outputs_dir = app.config['OUTPUT_FOLDER']
        if outputs_dir.exists():
            subdirs = [d.name for d in outputs_dir.iterdir() if d.is_dir()]
            print(f"📂 outputs 폴더 내 디렉토리: {subdirs}")
        else:
            print(f"⚠️ outputs 폴더가 존재하지 않음: {outputs_dir}")
        
        if not result_path.exists():
            error_msg = f'분석 결과를 찾을 수 없습니다: {video_stem}'
            print(f"❌ {error_msg}")
            print("=" * 50)
            return jsonify({
                'success': False,
                'error': error_msg,
                'debug_info': {
                    'searched_path': str(result_path),
                    'available_dirs': subdirs if outputs_dir.exists() else []
                }
            }), 404

        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        print("🤖 VQA 수행 시작...")
        # VQA 수행
        gen = IntegratedReportGenerator(results, api_key=app.config['GEMINI_API_KEY'])
        response = gen.answer_question(question)
        print(f"💬 VQA 응답: {response.get('success', False)}")
        print("=" * 50)
        return jsonify(response)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ 서버 오류:\n{error_trace}")
        print("=" * 50)
        return jsonify({
            'success': False, 
            'error': f'서버 오류: {str(e)}',
            'traceback': error_trace
        }), 500


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


def print_routes():
    """등록된 라우트 출력"""
    print("\n" + "=" * 50)
    print("📋 등록된 라우트:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
        print(f"  {rule.rule:30s} [{methods}]")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    print("\n🚀 Flask 서버 시작")
    print(f"📂 Upload 폴더: {app.config['UPLOAD_FOLDER']}")
    print(f"📂 Output 폴더: {app.config['OUTPUT_FOLDER']}")
    
    print_routes()
    
    # ✅ use_reloader=False로 설정하여 자동 재시작 방지
    # 개발 중에는 debug=False로 설정
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)