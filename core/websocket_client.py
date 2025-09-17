import asyncio
import websockets
from websockets.exceptions import ConnectionClosed
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from config.settings import NETWORK_CONFIG, COMMAND_CONFIG


class WebSocketCommandClient(QObject):
    """WebSocket client for sending commands only."""
    
    connection_status = pyqtSignal(bool, str)  # connected, message
    command_sent = pyqtSignal(str, bool)  # command, success
    
    def __init__(self):
        super().__init__()
        self.websocket = None
        self.running = False
        self.connected = False
        self.retry_count = 0
        self.max_retries = COMMAND_CONFIG['retry_attempts']
    
    async def connect_and_maintain(self):
        """Connect to WebSocket server and maintain connection."""
        websocket_ip = NETWORK_CONFIG['websocket_ip']
        websocket_port = NETWORK_CONFIG['websocket_port']
        uri = f"ws://{websocket_ip}:{websocket_port}"
        
        while self.running:
            try:
                print(f"Connecting to command WebSocket {uri}...")
                self.websocket = await websockets.connect(
                    uri, 
                    open_timeout=NETWORK_CONFIG['connection_timeout']
                )
                self.connected = True
                self.retry_count = 0
                self.connection_status.emit(True, "Command WebSocket connected")
                print(f"Connected to command WebSocket {uri}")
                
                # Keep connection alive
                await self.maintain_connection()
                
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                self.connected = False
                self.connection_status.emit(False, f"Connection error: {e}")
                
                if self.running:
                    self.retry_count += 1
                    if self.retry_count <= self.max_retries:
                        print(f"Retrying connection ({self.retry_count}/{self.max_retries})...")
                        await asyncio.sleep(NETWORK_CONFIG['reconnect_interval'])
                    else:
                        print("Max retries reached, stopping connection attempts")
                        break
            finally:
                if self.websocket:
                    await self.websocket.close()
                self.connected = False
                self.connection_status.emit(False, "Disconnected")
    
    async def maintain_connection(self):
        """Maintain WebSocket connection with periodic pings."""
        try:
            while self.running and self.connected:
                await asyncio.sleep(10)  # Ping every 10 seconds
                if self.websocket:
                    try:
                        # For websockets 15.0+, connection is automatically managed
                        # Just check if we can ping without accessing closed attribute
                        await asyncio.wait_for(self.websocket.ping(), timeout=5)
                    except (ConnectionClosed, asyncio.TimeoutError) as e:
                        print(f"Connection lost during ping: {e}")
                        break
                    except Exception as ping_error:
                        print(f"Ping failed: {ping_error}")
                        break
                else:
                    break
        except Exception as e:
            print(f"Connection maintenance error: {e}")
            self.connected = False
    
    async def send_command(self, command):
        """Send command through WebSocket."""
        if not self.websocket or not self.connected:
            print("WebSocket not connected, cannot send command")
            self.command_sent.emit(command, False)
            return False
        
        try:
            # For websockets 15.0+, just try to send - connection management is automatic
            # Validate command first
            if not self.validate_command(command):
                print(f"Invalid command: {command}")
                self.command_sent.emit(command, False)
                return False
            
            # Send command with timeout
            await asyncio.wait_for(
                self.websocket.send(command),
                timeout=COMMAND_CONFIG['command_timeout']
            )
            
            print(f"Sent command: {command}")
            self.command_sent.emit(command, True)
            return True
            
        except ConnectionClosed as e:
            print(f"WebSocket connection closed: {e}")
            self.connected = False
            self.connection_status.emit(False, "Connection closed")
            self.command_sent.emit(command, False)
            return False
        except asyncio.TimeoutError:
            print(f"Command timeout: {command}")
            self.command_sent.emit(command, False)
            return False
        except Exception as e:
            print(f"Error sending command: {e}")
            self.connected = False
            self.connection_status.emit(False, f"Send error: {e}")
            self.command_sent.emit(command, False)
            return False
    
    def validate_command(self, command):
        """Validate command format and content."""
        if not command or not isinstance(command, str):
            return False
        
        # Check if command starts with known command types
        command_lower = command.lower().strip()
        valid_commands = COMMAND_CONFIG['available_commands']
        
        for valid_cmd in valid_commands:
            if command_lower.startswith(valid_cmd):
                return True
        
        return False
    
    def start_client(self):
        """Start the WebSocket client."""
        self.running = True
        self.retry_count = 0
    
    def stop_client(self):
        """Stop the WebSocket client."""
        self.running = False
        self.connected = False


class WebSocketCommandThread(QThread):
    """Thread to run WebSocket command client."""
    
    def __init__(self, websocket_client):
        super().__init__()
        self.websocket_client = websocket_client
        self.loop = None
    
    def run(self):
        """Run the WebSocket client in async event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(
                self.websocket_client.connect_and_maintain()
            )
        except Exception as e:
            print(f"WebSocket thread error: {e}")
        finally:
            if self.loop and not self.loop.is_closed():
                self.loop.close()
    
    def send_command_sync(self, command):
        """Send command synchronously from main thread."""
        if not self.loop or self.loop.is_closed():
            print("Event loop not available")
            return False
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.websocket_client.send_command(command), 
                self.loop
            )
            return future.result(timeout=COMMAND_CONFIG['command_timeout'])
        except Exception as e:
            print(f"Error sending command sync: {e}")
            return False
    
    def stop(self):
        """Stop the WebSocket thread."""
        self.websocket_client.stop_client()
        self.quit()
        self.wait()