#!/usr/bin/env python3
"""
Auto-Sync: Real-time GitHub synchronization
Watches the core folder and automatically commits changes as you type
"""

import time
import os
import subprocess
import hashlib
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
from datetime import datetime

class AutoSyncHandler(FileSystemEventHandler):
    def __init__(self, core_path, git_path):
        self.core_path = core_path
        self.git_path = git_path
        self.file_hashes = {}
        self.last_commit_time = 0
        self.commit_delay = 5  # Wait 5 seconds after last change before committing
        self.pending_changes = set()
        
        # Load existing file hashes
        self.load_file_hashes()
        
    def load_file_hashes(self):
        """Load existing file hashes to detect changes"""
        for file_path in Path(self.core_path).rglob('*'):
            if file_path.is_file() and not str(file_path).startswith('.'):
                self.file_hashes[str(file_path)] = self.get_file_hash(file_path)
    
    def get_file_hash(self, file_path):
        """Get MD5 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        if not file_path.startswith(self.core_path):
            return
            
        # Skip hidden files and system files
        if os.path.basename(file_path).startswith('.'):
            return
            
        print(f"📝 File changed: {os.path.relpath(file_path, self.core_path)}")
        self.pending_changes.add(file_path)
        
        # Schedule commit after delay
        self.schedule_commit()
    
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        if not file_path.startswith(self.core_path):
            return
            
        print(f"🆕 File created: {os.path.relpath(file_path, self.core_path)}")
        self.pending_changes.add(file_path)
        self.schedule_commit()
    
    def on_deleted(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        if not file_path.startswith(self.core_path):
            return
            
        print(f"🗑️  File deleted: {os.path.relpath(file_path, self.core_path)}")
        self.pending_changes.add(file_path)
        self.schedule_commit()
    
    def schedule_commit(self):
        """Schedule a commit after the delay period"""
        current_time = time.time()
        self.last_commit_time = current_time
        
        # Schedule commit after delay
        def delayed_commit():
            time.sleep(self.commit_delay)
            if time.time() - self.last_commit_time >= self.commit_delay:
                self.commit_changes()
        
        import threading
        threading.Timer(self.commit_delay, delayed_commit).start()
    
    def commit_changes(self):
        """Commit pending changes to GitHub"""
        if not self.pending_changes:
            return
            
        try:
            print(f"\n🔄 Committing {len(self.pending_changes)} changes...")
            
            # Change to git directory
            os.chdir(self.git_path)
            
            # Add all changes in core folder
            result = subprocess.run(['git', 'add', 'core/'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Git add failed: {result.stderr}")
                return
            
            # Check if there are actual changes
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            if not result.stdout.strip():
                print("ℹ️  No changes to commit")
                self.pending_changes.clear()
                return
            
            # Create commit message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_msg = f"Auto-sync: {len(self.pending_changes)} files updated at {timestamp}"
            
            # Commit changes
            result = subprocess.run(['git', 'commit', '-m', commit_msg], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Git commit failed: {result.stderr}")
                return
            
            # Push to GitHub
            result = subprocess.run(['git', 'push'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Git push failed: {result.stderr}")
                return
            
            print(f"✅ Successfully committed and pushed to GitHub!")
            print(f"📝 Commit message: {commit_msg}")
            
            # Update file hashes and clear pending changes
            self.load_file_hashes()
            self.pending_changes.clear()
            
        except Exception as e:
            print(f"❌ Error during commit: {e}")
        finally:
            # Return to original directory
            os.chdir(self.core_path)

def main():
    print("🚀 Auto-Sync: Real-time GitHub synchronization")
    print("=" * 50)
    
    # Get current directory and core folder
    current_dir = os.getcwd()
    core_path = os.path.join(current_dir, 'core')
    git_path = current_dir
    
    if not os.path.exists(core_path):
        print(f"❌ Core folder not found: {core_path}")
        return
    
    if not os.path.exists(os.path.join(git_path, '.git')):
        print(f"❌ Git repository not found in: {git_path}")
        return
    
    print(f"📁 Watching: {core_path}")
    print(f"🔗 Git repo: {git_path}")
    print(f"⏱️  Commit delay: 5 seconds")
    print(f"🔄 Auto-sync is running... Press Ctrl+C to stop")
    print("=" * 50)
    
    # Create event handler and observer
    event_handler = AutoSyncHandler(core_path, git_path)
    observer = Observer()
    observer.schedule(event_handler, core_path, recursive=True)
    
    try:
        observer.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping auto-sync...")
        observer.stop()
        observer.join()
        print("✅ Auto-sync stopped")

if __name__ == "__main__":
    main()
