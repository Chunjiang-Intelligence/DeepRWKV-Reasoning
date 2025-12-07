from flask import Flask, render_template
from flask_socketio import SocketIO
import threading

class VisualizationServer:
    def __init__(self):
        self.app = Flask(__name__, template_folder='templates')
        self.socketio = SocketIO(self.app, async_mode='eventlet')
        self.app.add_url_rule('/', 'index', self.index)

    def index(self):
        return render_template('index.html')

    def update_tree_data(self, data):
        self.socketio.emit('update_tree', data)

    def run(self):
        server_thread = threading.Thread(target=lambda: self.socketio.run(self.app, host='0.0.0.0', port=5001))
        server_thread.daemon = True
        server_thread.start()
        print("[Visualizer] Server running at http://localhost:5001")