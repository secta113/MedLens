import os
# 【重要】PaddleOCRのネットワーク接続チェックをスキップ（起動高速化・エラー回避）
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# 【重要】PaddleOCRのデバッグログを抑制（引数ではなく環境変数で制御）
os.environ["GLOG_minloglevel"] = "2"

import flet as ft
import cv2
import threading
import base64
import time
from datetime import datetime
from paddleocr import PaddleOCR
import logging

# PaddleOCRの不要なログをPython側でも抑制
logging.getLogger("ppocr").setLevel(logging.ERROR)

# 初期化用ダミー画像
DUMMY_IMG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

# --- OCRクラス ---
class ReceiptOCR:
    def __init__(self):
        print("OCRエンジン初期化中... (モデルのロードを行います)")
        # 【修正】警告回避のため use_angle_cls=True から use_textline_orientation=True に変更
        self.ocr = PaddleOCR(use_textline_orientation=True, lang="japan")
        print("OCRエンジン準備完了")

    def analyze(self, image):
        # cls=True で文字の回転も補正して読み取ります
        result = self.ocr.ocr(image, cls=True)
        parsed_lines = []
        
        # 結果がNoneの場合のガード処理
        if result is None:
            return []

        # result構造: [ [ [ [x,y], ... ], ("text", conf) ], ... ]
        for line in result:
            if line is None: continue
            for word_info in line:
                if word_info is None: continue
                text = word_info[1][0]
                confidence = word_info[1][1]
                parsed_lines.append(f"{text} ({confidence:.2f})")
        return parsed_lines

ocr_engine = None

class AppState:
    def __init__(self):
        self.camera_running = True
        self.detected_doc = None
        self.canny_th1 = 50.0
        self.canny_th2 = 150.0
        self.min_area = 10000.0
        self.debug_mode = False 
        # デバッグ用フラグ
        self.force_page_update = False
        self.use_gapless = True

state = AppState()

def main(page: ft.Page):
    global ocr_engine
    
    page.title = "MedLens - Stable OCR"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.width = 1200
    page.window.height = 900
    
    # OCRエンジンの初期化
    if ocr_engine is None:
        try:
            ocr_engine = ReceiptOCR()
        except Exception as e:
            print(f"OCR初期化エラー: {e}")

    # --- デバッグ表示用コントロール ---
    fps_text = ft.Text("UI FPS: 0", color=ft.Colors.RED, size=20, weight="bold")
    last_render_text = ft.Text("Last Render: -", size=12, color=ft.Colors.GREY)
    heartbeat_text = ft.Text("UI Heartbeat: -", size=16, color=ft.Colors.BLUE)

    img_control = ft.Image(
        src=DUMMY_IMG,
        width=640,
        height=480,
        fit="contain",
        gapless_playback=state.use_gapless,
    )
    
    status_text = ft.Text("カメラ待機中...")
    log_column = ft.Column(scroll=ft.ScrollMode.ALWAYS, height=200)

    def log(msg):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_column.controls.insert(0, ft.Text(f"[{timestamp}] {msg}"))
        log_column.update()

    # --- イベントハンドラ ---
    def on_th1_change(e): state.canny_th1 = float(e.control.value)
    def on_th2_change(e): state.canny_th2 = float(e.control.value)
    def on_area_change(e): state.min_area = float(e.control.value)
    def on_debug_switch(e): state.debug_mode = e.control.value

    # デバッグ設定の切り替え
    def on_force_update_change(e):
        state.force_page_update = e.control.value
        log(f"強制ページ更新: {state.force_page_update}")

    def on_gapless_change(e):
        state.use_gapless = e.control.value
        img_control.gapless_playback = state.use_gapless
        img_control.update()
        log(f"Gapless Playback: {state.use_gapless}")

    def on_capture_click(e):
        if state.detected_doc is None:
            log("エラー: 書類が検出されていません")
            return
        
        if ocr_engine is None:
            log("エラー: OCRエンジンが初期化されていません")
            return

        log("OCR解析を開始します...")
        try:
            target_img = state.detected_doc.copy()
            lines = ocr_engine.analyze(target_img)
            log("--- 解析結果 ---")
            if not lines:
                log("文字が見つかりませんでした")
            else:
                for line in lines:
                    log(line)
            log("----------------")
        except Exception as err:
            log(f"OCR実行時エラー: {err}")
            print(err) # 詳細をコンソールにも出す

    # --- UI生存確認用スレッド（ハートビート） ---
    def heartbeat_loop():
        while state.camera_running:
            current_time = datetime.now().strftime('%H:%M:%S')
            heartbeat_text.value = f"UI Heartbeat: {current_time} (Alive)"
            try:
                # ハートビートは常に強制更新しないと意味がないので page.update は使わない
                # （ただしメインスレッドが止まるとこれも止まります）
                heartbeat_text.update()
            except:
                pass
            time.sleep(1.0)

    # --- カメラ処理ループ ---
    def camera_loop():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        last_ui_update_time = 0
        frame_count = 0
        UI_UPDATE_INTERVAL = 0.05 # 20FPS
        
        while state.camera_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # 検出処理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, int(state.canny_th1), int(state.canny_th2))
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected = False
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                area = cv2.contourArea(c)
                
                if len(approx) == 4 and area > state.min_area:
                    # OCR精度向上のため、緑の輪郭線を描画する前のクリーンな画像を保存
                    state.detected_doc = frame.copy()
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                    detected = True

            # 画面更新タイミング制御
            current_time = time.time()
            if current_time - last_ui_update_time < UI_UPDATE_INTERVAL:
                time.sleep(0.001)
                continue

            # 画像生成
            display_frame = edged if state.debug_mode else frame
            display_h, display_w = display_frame.shape[:2]
            if display_w > 640:
                scale = 640 / display_w
                small_frame = cv2.resize(display_frame, (0, 0), fx=scale, fy=scale)
            else:
                small_frame = display_frame

            _, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            
            if img_control.page is None:
                time.sleep(0.1)
                continue
            
            # --- 更新処理 ---
            img_control.src = f"data:image/jpeg;base64,{b64_str}"
            
            # デバッグ情報更新
            now_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            last_render_text.value = f"Last Render Request: {now_str}"
            
            frame_count += 1
            fps_text.value = f"UI Update Count: {frame_count}"

            # ステータス情報の更新判定
            new_status_val = "検出中！" if detected else "..."
            new_status_color = ft.Colors.GREEN if detected else ft.Colors.BLACK
            
            status_changed = False
            if status_text.value != new_status_val or status_text.color != new_status_color:
                status_text.value = new_status_val
                status_text.color = new_status_color
                status_changed = True

            # 更新モードの分岐
            if state.force_page_update:
                page.update()
            else:
                img_control.update()
                fps_text.update()
                last_render_text.update()
                if status_changed:
                    status_text.update()

            last_ui_update_time = current_time
        
        cap.release()

    # --- UI構築 ---
    slider_th1 = ft.Slider(min=0, max=255, divisions=255, value=state.canny_th1, label="エッジ検出 しきい値1: {value}", on_change=on_th1_change)
    slider_th2 = ft.Slider(min=0, max=255, divisions=255, value=state.canny_th2, label="エッジ検出 しきい値2: {value}", on_change=on_th2_change)
    slider_area = ft.Slider(min=1000, max=100000, divisions=100, value=state.min_area, label="最小面積: {value}", on_change=on_area_change)
    switch_debug = ft.Switch(label="デバッグ表示（機械の目）", value=False, on_change=on_debug_switch)
    
    switch_force_update = ft.Switch(label="強制ページ更新 (Page Update)", value=False, on_change=on_force_update_change)
    switch_gapless = ft.Switch(label="Gapless Playback有効", value=True, on_change=on_gapless_change)

    page.add(
        ft.Row([
            ft.Column([
                ft.Text("カメラ映像", size=20, weight="bold"),
                ft.Row([fps_text, heartbeat_text], spacing=20),
                ft.Container(
                    content=img_control,
                    border=ft.Border.all(1, ft.Colors.GREY_400),
                ),
                last_render_text
            ]),
            ft.Column([
                ft.Text("操作・設定パネル", size=20, weight="bold"),
                status_text,
                ft.Button(
                    content=ft.Text("手動キャプチャ＆OCR解析"),
                    on_click=on_capture_click,
                    style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_50, color=ft.Colors.BLUE)
                ),
                ft.Divider(),
                
                ft.Text("▼ レンダリング設定", weight="bold", color=ft.Colors.RED),
                switch_force_update,
                switch_gapless,
                
                ft.Divider(),
                ft.Text("認識パラメータ"),
                switch_debug,
                ft.Text("しきい値1"), slider_th1,
                ft.Text("しきい値2"), slider_th2,
                ft.Text("最小面積"), slider_area,
                
                ft.Divider(),
                ft.Text("システムログ"),
                ft.Container(
                    content=log_column,
                    border=ft.Border.all(1, ft.Colors.GREY_300),
                    padding=10,
                    width=400,
                    height=250
                )
            ], expand=True)
        ], expand=True)
    )

    # スレッド開始
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=heartbeat_loop, daemon=True).start()

if __name__ == "__main__":
    ft.run(main)