# Banking Anomaly Detection System - Project Architecture (Updated)

## 🏗 **System Architecture Overview**

The Banking Anomaly Detection System follows a **layered microservices architecture** with enhanced ML capabilities, featuring dual neural networks, centralized configuration, and production-ready design for robust fraud detection.

## 📐 **Enhanced High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    🌐 PRESENTATION LAYER                    │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Web Interface (app.py)                          │
│  ├── Authentication & Session Management                    │
│  ├── Enhanced Dashboard & Visualization                    │
│  ├── Transaction Input Forms                               │
│  └── Detailed Results Display & Analytics                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   🧠 BUSINESS LOGIC LAYER                   │
├─────────────────────────────────────────────────────────────┤
│  Enhanced Hybrid Decision Engine (hybrid_decision.py)      │
│  ├── 🚫 Rule Engine Integration                            │
│  ├── 🌲 Isolation Forest Integration                       │
│  ├── 🧠 Autoencoder Neural Network Integration             │
│  └── 🎯 Advanced Decision Aggregation Logic                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                🔍 ENHANCED DETECTION SERVICES LAYER         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │🚫 Rule      │  │🌲 Isolation │  │🧠 Autoencoder│        │
│  │  Engine     │  │  Forest     │  │  Neural Net │        │
│  │             │  │             │  │             │        │
│  │• Velocity   │  │• Anomaly    │  │• Behavioral │        │
│  │• Limits     │  │  Detection  │  │  Analysis   │        │
│  │• Thresholds │  │• Risk Score │  │• Pattern    │        │
│  │             │  │             │  │  Learning   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         🔄 Training/Inference Separation            │  │
│  │  ┌─────────────┐              ┌─────────────┐      │  │
│  │  │🏋️ Training   │              │⚡ Inference  │      │  │
│  │  │  Services   │              │  Services   │      │  │
│  │  │             │              │             │      │  │
│  │  │• Model      │              │• Fast       │      │  │
│  │  │  Training   │              │  Prediction │      │  │
│  │  │• Feature    │              │• Real-time  │      │  │
│  │  │  Learning   │              │  Scoring    │      │  │
│  │  └─────────────┘              └─────────────┘      │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                🔧 ENHANCED DATA PROCESSING LAYER            │
├─────────────────────────────────────────────────────────────┤
│  Enhanced Feature Engineering (feature_engineering.py)     │
│  ├── 41 Advanced Features (vs 26 previously)               │
│  ├── Weekly/Monthly Spending Analytics                     │
│  ├── Enhanced Behavioral Patterns                          │
│  ├── Cross-Account Transfer Analysis                       │
│  ├── Beneficiary Relationship Tracking                     │
│  └── Centralized Feature Configuration (utils.py)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                💾 ENHANCED DATA STORAGE LAYER               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │📊 Enhanced  │  │🤖 Dual ML   │  │⚙️ Centralized│        │
│  │Training Data│  │   Models    │  │Configuration│        │
│  │             │  │             │  │             │        │
│  │• 41 Features│  │• isolation_ │  │• MODEL_     │        │
│  │• Weekly/    │  │  forest.pkl │  │  FEATURES   │        │
│  │  Monthly    │  │• autoencoder│  │• Thresholds │        │
│  │  Analytics  │  │  .h5        │  │• Scalers    │        │
│  │• Behavioral │  │• Scalers    │  │• Paths      │        │
│  │  Patterns   │  │• Thresholds │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  📁 Organized Model Storage: backend/model/                │
└─────────────────────────────────────────────────────────────┘
```

## 🧩 **Enhanced Component Architecture**

### **1. Enhanced Presentation Layer**

#### **Streamlit Web Application (app.py) - Updated**
```
🌐 Enhanced Web Interface Architecture
├── 🔐 Authentication Module
│   ├── Session management
│   ├── User validation
│   └── Security controls
├── 📊 Enhanced Dashboard Components
│   ├── Dual ML model results display
│   ├── 41-feature analysis visualization
│   ├── Real-time processing with detailed metrics
│   ├── Individual model score breakdown
│   └── Enhanced system status monitoring
├── 🎨 Improved UI/UX Elements
│   ├── Responsive design with better layouts
│   ├── Interactive charts for both models
│   ├── Detailed explanation panels
│   └── User-friendly navigation with tooltips
└── 📈 Advanced Analytics Display
    ├── Isolation Forest anomaly scores
    ├── Autoencoder reconstruction errors
    ├── Combined risk assessment
    └── Feature importance visualization
```

### **2. Enhanced Business Logic Layer**

#### **Enhanced Hybrid Decision Engine (hybrid_decision.py)**
```
🎯 Advanced Decision Integration Architecture
├── 🔄 Enhanced Processing Pipeline
│   ├── Sequential layer execution with logging
│   ├── Priority-based decision making
│   ├── Detailed result aggregation
│   └── Performance monitoring
├── 🚫 Rule Engine Interface (Unchanged)
│   ├── Business rule validation
│   ├── Hard limit enforcement
│   └── Immediate blocking logic
├── 🌲 Enhanced ML Model Interface
│   ├── Isolation Forest integration
│   ├── Anomaly score processing
│   ├── Risk assessment with confidence scores
│   └── Feature importance analysis
└── 🧠 New Neural Network Interface
    ├── Autoencoder integration
    ├── Behavioral pattern analysis
    ├── Reconstruction error evaluation
    └── Behavioral anomaly detection
```

### **3. Enhanced Detection Services Layer**

#### **Training Services (New Architecture)**
```
🏋️ Training Services Architecture
├── 📚 IsolationForestTrainer Class
│   ├── Enhanced training pipeline
│   ├── 41-feature processing
│   ├── Model validation and metrics
│   └── Automated threshold calculation
├── 🧠 AutoencoderTrainer Class
│   ├── Neural network architecture design
│   ├── Training with early stopping
│   ├── Reconstruction error analysis
│   └── Statistical threshold computation
└── 🔧 Shared Training Infrastructure
    ├── Centralized feature loading
    ├── Consistent data preprocessing
    ├── Model persistence management
    └── Training metrics collection
```

#### **Inference Services (New Architecture)**
```
⚡ Inference Services Architecture
├── 🌲 IsolationForestInference Class
│   ├── Fast model loading
│   ├── Real-time anomaly scoring
│   ├── Feature scaling and validation
│   └── Risk score calculation
├── 🧠 AutoencoderInference Class
│   ├── Neural network inference
│   ├── Reconstruction error computation
│   ├── Behavioral anomaly detection
│   └── Pattern analysis results
└── 🚀 Optimized Performance
    ├── Model caching strategies
    ├── Batch processing capabilities
    ├── Memory-efficient operations
    └── Sub-100ms response times
```

#### **Enhanced Autoencoder Service (New)**
```
🧠 Advanced Neural Network Architecture
├── 🏗 Enhanced Model Structure
│   ├── Input Layer: 41 features (vs 26 previously)
│   ├── Encoder: [64, 32] → Bottleneck (adaptive size)
│   ├── Decoder: Bottleneck → [32, 64] → 41 features
│   ├── Batch Normalization layers
│   └── Advanced activation functions
├── 🔧 Enhanced Training Pipeline
│   ├── Advanced data preprocessing
│   ├── Early stopping with patience
│   ├── Learning rate scheduling
│   ├── Model checkpointing
│   └── Comprehensive validation
├── ⚡ Production Inference Engine
│   ├── Optimized model loading
│   ├── Real-time reconstruction
│   ├── Statistical error analysis
│   ├── Behavioral pattern recognition
│   └── Confidence score calculation
└── 📊 Advanced Analytics
    ├── Feature reconstruction quality
    ├── Anomaly confidence levels
    ├── Behavioral change detection
    └── Pattern evolution tracking
```

## 🗂 **Enhanced File Structure Architecture**

```
banking_anomaly_detector/
├── 📱 Frontend Layer
│   └── app.py                          # Enhanced Streamlit interface
├── 🧠 Enhanced Business Logic Layer  
│   └── backend/
│       ├── hybrid_decision.py          # Enhanced decision integration
│       ├── rule_engine.py              # Business rules (unchanged)
│       ├── train_isolation_forest.py   # IF training service
│       ├── isolation_forest.py         # IF inference service
│       ├── train_autoencoder.py        # AE training service
│       ├── autoencoder.py              # AE inference service
│       ├── feature_engineering.py     # Enhanced 41-feature processing
│       └── utils.py                    # Centralized config (MODEL_FEATURES)
├── 💾 Enhanced Data Layer
│   ├── data/                           # Enhanced training datasets
│   │   ├── feature_datasetv2.csv      # 41-feature dataset
│   │   └── Clean.csv                   # Original data
│   └── backend/model/                  # Organized model storage
│       ├── isolation_forest.pkl        # IF model
│       ├── isolation_forest_scaler.pkl # IF scaler
│       ├── autoencoder.h5             # AE model
│       ├── autoencoder_scaler.pkl     # AE scaler
│       └── autoencoder_threshold.json  # AE configuration
├── 🧪 Enhanced Testing Layer
│   └── tests/
│       ├── test_autoencoder_properties.py
│       ├── test_autoencoder_errors.py
│       └── test_frontend_ae.py
└── 📚 Organized Documentation
    └── docs/
        ├── README.md                   # Documentation index
        ├── BRD.md                     # Enhanced business requirements
        ├── PROJECT_LOGIC.md           # Updated system logic
        ├── projectarchitecture.md    # This document
        └── projectflow.md             # Enhanced workflow
```

## 🔄 **Enhanced Data Flow Architecture**

### **Enhanced Training Data Flow**
```
📊 Advanced Training Pipeline
Raw Data → Enhanced Feature Engineering → Dual Model Training → Organized Storage
    ↓              ↓                           ↓                    ↓
CSV Files → 41 Features → IF + AE Models → backend/model/ directory
                ↓              ↓
        Centralized Config  Statistical Thresholds
```

### **Enhanced Inference Data Flow**
```
⚡ Real-time Processing (Enhanced)
Transaction → 41 Features → Rule Check → IF Analysis → AE Analysis → Combined Decision
     ↓           ↓            ↓           ↓            ↓              ↓
  Input Data → Enhanced → Block/Pass → Anomaly → Behavioral → Detailed Result
              Features              Score     Analysis
```

## 🛡 **Enhanced Security Architecture**

### **Advanced Data Protection**
```
🔒 Enhanced Security Layers
├── 🔐 Authentication
│   ├── Session-based login
│   ├── User validation
│   └── Access control
├── 🛡 Enhanced Data Security
│   ├── Input validation for 41 features
│   ├── Model integrity verification
│   ├── SQL injection prevention
│   └── XSS protection
├── 🔍 Advanced Audit Logging
│   ├── Dual model decision tracking
│   ├── Feature-level analysis logs
│   ├── User activity monitoring
│   └── System performance metrics
└── 🚨 Anomaly Detection Security
    ├── Model tampering detection
    ├── Feature manipulation alerts
    └── Inference integrity checks
```

## ⚡ **Enhanced Performance Architecture**

### **Advanced Optimization Strategies**
```
🚀 Enhanced Performance Design
├── 💾 Advanced Caching Layer
│   ├── Dual model caching (@st.cache_resource)
│   ├── Feature preprocessing caching
│   ├── Scaler caching for both models
│   └── Threshold configuration caching
├── 🔄 Optimized Processing
│   ├── Lazy model loading
│   ├── Batch feature computation
│   ├── Memory-efficient operations
│   └── CPU/GPU optimization
├── 📊 Enhanced Scalability
│   ├── Stateless inference design
│   ├── Horizontal scaling ready
│   ├── Load balancing support
│   └── Microservices architecture
└── 🎯 Performance Monitoring
    ├── Real-time latency tracking
    ├── Memory usage optimization
    ├── Model performance metrics
    └── System resource monitoring
```

## 🔧 **Enhanced Technology Stack**

### **Advanced Core Technologies**
```
🛠 Enhanced Technology Architecture
├── 🐍 Backend Framework
│   ├── Python 3.13
│   ├── Streamlit (Enhanced Web UI)
│   ├── NumPy/Pandas (Advanced data processing)
│   └── Joblib (Model persistence)
├── 🤖 Advanced Machine Learning
│   ├── Scikit-learn (Isolation Forest)
│   ├── TensorFlow/Keras (Autoencoder Neural Network)
│   ├── StandardScaler (Feature normalization)
│   └── Advanced preprocessing pipelines
├── 💾 Enhanced Data Storage
│   ├── CSV files (Enhanced training data)
│   ├── PKL files (ML models + scalers)
│   ├── H5 files (Neural network models)
│   └── JSON files (Configuration + thresholds)
├── 🧪 Advanced Testing & Quality
│   ├── Hypothesis (Property-based testing)
│   ├── Pytest (Unit testing)
│   ├── Custom validation frameworks
│   └── Model performance testing
└── 📊 Analytics & Monitoring
    ├── Performance metrics collection
    ├── Model accuracy tracking
    ├── Feature importance analysis
    └── System health monitoring
```

## 🔌 **Enhanced Integration Architecture**

### **Advanced External System Integration**
```
🔗 Enhanced Integration Points
├── 📊 Advanced Data Sources
│   ├── Real-time transaction streams
│   ├── Historical behavior databases
│   ├── Cross-account relationship data
│   └── Beneficiary pattern analysis
├── 🚨 Enhanced Alerting Systems
│   ├── Multi-model fraud notifications
│   ├── Behavioral anomaly alerts
│   ├── System performance monitoring
│   └── Model drift detection
├── 📈 Advanced Analytics Platforms
│   ├── Enhanced business intelligence
│   ├── Dual-model reporting systems
│   ├── Compliance tracking
│   └── Performance analytics
└── 🔄 Real-time Processing
    ├── Stream processing capabilities
    ├── Event-driven architecture
    ├── Microservices communication
    └── API gateway integration
```

## 🎯 **Enhanced Deployment Architecture**

### **Production-Ready Environment Strategy**
```
🚀 Enhanced Deployment Design
├── 🧪 Development Environment
│   ├── Local development with dual models
│   ├── Enhanced unit testing
│   ├── Feature development sandbox
│   └── Model experimentation platform
├── 🔍 Testing Environment
│   ├── Integration testing for both models
│   ├── Performance testing with 41 features
│   ├── User acceptance testing
│   └── Model accuracy validation
├── 🏭 Production Environment
│   ├── High availability setup
│   ├── Load balancing for inference services
│   ├── Advanced monitoring & alerting
│   ├── Automated backup & recovery
│   └── Model versioning and rollback
└── 🔄 CI/CD Pipeline
    ├── Automated model training
    ├── Model validation gates
    ├── Performance regression testing
    └── Automated deployment
```

## 📊 **Model Performance Architecture**

### **Dual Model Monitoring**
```
📈 Enhanced Performance Tracking
├── 🌲 Isolation Forest Metrics
│   ├── Anomaly detection accuracy
│   ├── False positive/negative rates
│   ├── Processing latency
│   └── Feature importance tracking
├── 🧠 Autoencoder Metrics
│   ├── Reconstruction error distribution
│   ├── Behavioral anomaly detection rate
│   ├── Neural network performance
│   └── Pattern recognition accuracy
├── 🎯 Combined System Metrics
│   ├── Overall fraud detection rate
│   ├── System throughput
│   ├── Decision confidence levels
│   └── User experience metrics
└── 🔄 Continuous Improvement
    ├── Model retraining triggers
    ├── Feature engineering optimization
    ├── Threshold adjustment automation
    └── Performance optimization cycles
```

This enhanced architecture provides enterprise-grade fraud detection with state-of-the-art machine learning, clean separation of concerns, and production-ready scalability while maintaining the flexibility to adapt and improve over time.