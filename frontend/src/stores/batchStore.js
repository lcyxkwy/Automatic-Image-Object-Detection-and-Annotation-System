import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// 批量处理状态管理
export const useBatchStore = defineStore('batch', () => {
  // 状态
  const results = ref([])
  const fileList = ref([])
  const processing = ref(false)
  const processedCount = ref(0)
  
  // 检测参数
  const detectParams = ref({
    weights: 'yolov5s.pt',
    confThreshold: 0.25,
    iouThreshold: 0.45,
    imgSize: 640
  })

  // 计算属性
  const totalDetections = computed(() => {
    return results.value.reduce((sum, r) => sum + (r.detections?.length || 0), 0)
  })

  const averageTime = computed(() => {
    if (results.value.length === 0) return 0
    const total = results.value.reduce((sum, r) => sum + (r.inference_time || 0), 0)
    return Math.round(total / results.value.length)
  })

  const totalTime = computed(() => {
    return results.value.reduce((sum, r) => sum + (r.inference_time || 0), 0)
  })

  // 方法
  const addResult = (result) => {
    results.value.push(result)
    saveToStorage()
  }

  const clearResults = () => {
    results.value = []
    fileList.value = []
    processedCount.value = 0
    saveToStorage()
  }

  const setProcessing = (value) => {
    processing.value = value
  }

  const setProcessedCount = (value) => {
    processedCount.value = value
  }

  const setDetectParams = (params) => {
    detectParams.value = { ...detectParams.value, ...params }
    saveToStorage()
  }

  const setFileList = (list) => {
    fileList.value = list
  }

  // 保存到 localStorage
  const saveToStorage = () => {
    try {
      const dataToSave = {
        results: results.value.map(r => ({
          ...r,
          file: null // 不保存 File 对象
        })),
        detectParams: detectParams.value
      }
      localStorage.setItem('batchStore', JSON.stringify(dataToSave))
    } catch (e) {
      console.error('保存批量处理状态失败:', e)
    }
  }

  // 从 localStorage 恢复
  const loadFromStorage = () => {
    try {
      const saved = localStorage.getItem('batchStore')
      if (saved) {
        const data = JSON.parse(saved)
        if (data.results) {
          results.value = data.results
        }
        if (data.detectParams) {
          detectParams.value = { ...detectParams.value, ...data.detectParams }
        }
      }
    } catch (e) {
      console.error('恢复批量处理状态失败:', e)
    }
  }

  // 初始化时加载
  loadFromStorage()

  return {
    // 状态
    results,
    fileList,
    processing,
    processedCount,
    detectParams,
    // 计算属性
    totalDetections,
    averageTime,
    totalTime,
    // 方法
    addResult,
    clearResults,
    setProcessing,
    setProcessedCount,
    setDetectParams,
    setFileList,
    saveToStorage,
    loadFromStorage
  }
})
