<template>
  <div class="welcome-page">
    <div class="welcome-container">
      <!-- 背景动画 -->
      <div class="bg-animation">
        <div class="particles">
          <span v-for="i in 50" :key="i" class="particle"></span>
        </div>
      </div>
      
      <!-- 主内容 -->
      <div class="welcome-content">
        <div class="logo-section">
          <img src="@/assets/logo.svg" alt="logo" class="main-logo" />
        </div>
        
        <h1 class="title">自动图像目标检测与标注系统</h1>
        <p class="subtitle">Automatic Image Object Detection and Annotation System</p>
        
        <div class="features">
          <div class="feature-item">
            <el-icon><Camera /></el-icon>
            <span>智能检测</span>
          </div>
          <div class="feature-item">
            <el-icon><Edit /></el-icon>
            <span>精准标注</span>
          </div>
          <div class="feature-item">
            <el-icon><DataLine /></el-icon>
            <span>模型训练</span>
          </div>
          <div class="feature-item">
            <el-icon><Files /></el-icon>
            <span>批量处理</span>
          </div>
        </div>
        
        <div class="tech-stack">
          <span class="tech-tag">YOLOv5</span>
          <span class="tech-tag">PyTorch</span>
          <span class="tech-tag">Vue 3</span>
          <span class="tech-tag">FastAPI</span>
        </div>
        
        <el-button 
          type="primary" 
          size="large" 
          class="enter-btn"
          @click="enterSystem"
        >
          <span>进入系统</span>
          <el-icon class="el-icon--right"><ArrowRight /></el-icon>
        </el-button>
        
        <div class="footer-info">
          <p>基于深度学习的目标检测与标注一体化解决方案</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { Camera, Edit, DataLine, Files, ArrowRight } from '@element-plus/icons-vue'

const router = useRouter()

const enterSystem = () => {
  // 记录已访问过欢迎页
  sessionStorage.setItem('hasVisited', 'true')
  router.push('/home')
}
</script>

<style lang="scss" scoped>
.welcome-page {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}

.welcome-container {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.bg-animation {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  z-index: 0;
  
  .particles {
    position: relative;
    width: 100%;
    height: 100%;
  }
  
  .particle {
    position: absolute;
    width: 4px;
    height: 4px;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    animation: float 15s infinite;
    
    @for $i from 1 through 50 {
      &:nth-child(#{$i}) {
        left: random(100) * 1%;
        top: random(100) * 1%;
        animation-delay: random(10) * -1s;
        animation-duration: 10s + random(10) * 1s;
        opacity: 0.1 + random(5) * 0.1;
        transform: scale(0.5 + random(10) * 0.1);
      }
    }
  }
}

@keyframes float {
  0%, 100% {
    transform: translateY(0) translateX(0);
    opacity: 0;
  }
  10% {
    opacity: 1;
  }
  90% {
    opacity: 1;
  }
  100% {
    transform: translateY(-100vh) translateX(20px);
    opacity: 0;
  }
}

.welcome-content {
  position: relative;
  z-index: 1;
  text-align: center;
  color: #fff;
  padding: 40px;
  animation: fadeInUp 1s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.logo-section {
  margin-bottom: 30px;
  
  .main-logo {
    width: 120px;
    height: 120px;
    filter: drop-shadow(0 0 20px rgba(64, 158, 255, 0.5));
    animation: pulse 2s infinite;
  }
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
}

.title {
  font-size: 42px;
  font-weight: 700;
  margin-bottom: 15px;
  background: linear-gradient(90deg, #fff, #a8d8ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: 0 0 30px rgba(64, 158, 255, 0.3);
}

.subtitle {
  font-size: 18px;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 40px;
  letter-spacing: 2px;
}

.features {
  display: flex;
  justify-content: center;
  gap: 40px;
  margin-bottom: 30px;
  
  .feature-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    
    .el-icon {
      font-size: 32px;
      color: #409EFF;
      padding: 15px;
      background: rgba(64, 158, 255, 0.1);
      border-radius: 50%;
      border: 1px solid rgba(64, 158, 255, 0.3);
      transition: all 0.3s;
      
      &:hover {
        transform: translateY(-5px);
        background: rgba(64, 158, 255, 0.2);
        box-shadow: 0 10px 30px rgba(64, 158, 255, 0.3);
      }
    }
    
    span {
      font-size: 14px;
      color: rgba(255, 255, 255, 0.8);
    }
  }
}

.tech-stack {
  display: flex;
  justify-content: center;
  gap: 15px;
  margin-bottom: 40px;
  flex-wrap: wrap;
  
  .tech-tag {
    padding: 6px 16px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    font-size: 13px;
    color: rgba(255, 255, 255, 0.8);
    transition: all 0.3s;
    
    &:hover {
      background: rgba(64, 158, 255, 0.2);
      border-color: rgba(64, 158, 255, 0.5);
    }
  }
}

.enter-btn {
  padding: 15px 50px;
  font-size: 18px;
  border-radius: 30px;
  background: linear-gradient(90deg, #409EFF, #66b1ff);
  border: none;
  box-shadow: 0 10px 30px rgba(64, 158, 255, 0.4);
  transition: all 0.3s;
  
  &:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(64, 158, 255, 0.5);
  }
  
  span {
    margin-right: 8px;
  }
}

.footer-info {
  margin-top: 50px;
  
  p {
    font-size: 14px;
    color: rgba(255, 255, 255, 0.5);
  }
}
</style>
