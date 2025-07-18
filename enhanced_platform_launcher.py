import streamlit as st
import sys
import os
import importlib

def main():
    st.set_page_config(page_title="NeuroInsight Diagnostics", layout="wide")
    st.title("🧠 NeuroInsight Platform Diagnostics")
    
    # Show current directory
    st.write("**Current Directory:**", os.getcwd())
    
    # List Python files
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    st.write("**Python Files Found:**")
    for file in sorted(py_files):
        size = os.path.getsize(file)
        st.write(f"- {file} ({size:,} bytes)")
    
    # Test imports
    st.subheader("Import Testing")
    
    modules_to_test = [
        'neural_simulation',
        'neuroinsight_africa_complete', 
        'main_neuromarketing_app'
    ]
    
    for module in modules_to_test:
        try:
            exec(f"import {module}")
            st.success(f"✅ {module} - Import successful")
        except Exception as e:
            st.error(f"❌ {module} - Error: {e}")
    
    # Platform launchers
    st.subheader("Platform Launch Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Launch Main App"):
            try:
                import main_neuromarketing_app
                st.success("Main app loaded")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("Launch Africa Platform"):
            try:
                import neuroinsight_africa_complete
                platform = neuroinsight_africa_complete.NeuroInsightAfrica()
                platform.run()
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col3:
        if st.button("Test Neural Simulation"):
            try:
                import neural_simulation
                st.success("Neural simulation loaded")
                st.write("Available attributes:", [attr for attr in dir(neural_simulation) if not attr.startswith('_')])
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

def load_enhanced_platform():
    st.set_page_config(
        page_title="Enhanced NeuroInsight Africa Platform",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Enhanced NeuroInsight Africa Platform")
    st.markdown("**Integrated African Market Intelligence + Advanced Neural Monitoring**")
    
    # Platform selector
    platform_mode = st.sidebar.selectbox(
        "Select Platform Mode:",
        [
            "Complete African Platform",
            "Neural Simulation Enhanced", 
            "African Market Intelligence",
            "Original Marketing App",
            "File Diagnostics"
        ]
    )
    
    try:
        if platform_mode == "Complete African Platform":
            import neuroinsight_africa_complete
            st.success("✅ Complete African Platform Module Loaded")
            platform = neuroinsight_africa_complete.NeuroInsightAfricaComplete()
            platform.run()
            
        elif platform_mode == "Neural Simulation Enhanced":
            import neural_simulation
            st.success("✅ Enhanced Neural Simulation Module Loaded")
            st.info("Neural simulation classes available")
            # Show available classes
            classes = [attr for attr in dir(neural_simulation) if attr[0].isupper()]
            st.write("Available Classes:", classes)
            
        elif platform_mode == "African Market Intelligence":
            try:
                import neuroinsight_africa_module
                st.success("✅ African Market Intelligence Module Loaded")
                # Show available functionality
                classes = [attr for attr in dir(neuroinsight_africa_module) if attr[0].isupper()]
                st.write("Available Classes:", classes)
            except ImportError:
                st.warning("African Market Intelligence module not found")
                
        elif platform_mode == "Original Marketing App":
            import main_neuromarketing_app
            st.success("✅ Original Marketing App Loaded")
            
        elif platform_mode == "File Diagnostics":
            st.subheader("📁 File System Diagnostics")
            
            # List all Python files
            py_files = [f for f in os.listdir('.') if f.endswith('.py')]
            st.write("**Available Python Files:**")
            for file in sorted(py_files):
                file_size = os.path.getsize(file)
                st.write(f"- {file} ({file_size:,} bytes)")
            
            # Test imports
            st.subheader("🔍 Import Testing")
            test_modules = [
                'neural_simulation',
                'neuroinsight_africa_complete', 
                'neuroinsight_africa_module',
                'main_neuromarketing_app'
            ]
            
            for module in test_modules:
                try:
                    importlib.import_module(module)
                    st.success(f"✅ {module} - Import successful")
                except ImportError as e:
                    st.error(f"❌ {module} - Import failed: {e}")
                except Exception as e:
                    st.warning(f"⚠️ {module} - Import error: {e}")
            
            # Git status
            st.subheader("📋 Git Status")
            try:
                import subprocess
                result = subprocess.run(['git', 'status', '--porcelain'], 
                                      capture_output=True, text=True)
                if result.stdout:
                    st.code(result.stdout)
                else:
                    st.info("Working directory clean")
                    
                # Show current branch
                branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                             capture_output=True, text=True)
                st.info(f"Current branch: {branch_result.stdout.strip()}")
                
            except Exception as e:
                st.error(f"Git command failed: {e}")
            
    except Exception as e:
        st.error(f"❌ Module loading error: {e}")
        st.code(str(e))
        
        # Show Python path for debugging
        st.subheader("🐍 Python Path Debug")
        st.code('\n'.join(sys.path))

if __name__ == "__main__":
    load_enhanced_platform()