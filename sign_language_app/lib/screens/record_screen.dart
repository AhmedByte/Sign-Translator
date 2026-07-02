import 'dart:async';
import 'package:camera/camera.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/prediction_service.dart';
import 'result_screen.dart';

enum RecordMode { select, camera }

class RecordScreen extends StatefulWidget {
  const RecordScreen({super.key});

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen>
    with TickerProviderStateMixin {
  RecordMode _currentMode = RecordMode.select;
  CameraController? _cameraController;
  bool _isInitialized = false;
  bool _isRecording = false;
  bool _isProcessing = false;
  String _processingMessage = 'جاري تحليل الإشارة...';
  Timer? _processingTimer;
  bool _processingIsCamera = false;
  int _currentStepIndex = 0;
  int _recordingSeconds = 0;
  Timer? _timer;
  bool _serverOnline = false;
  bool _checkingServer = true;
  late AnimationController _pulseController;

  void _startProcessingProgress(bool isCamera) {
    _processingTimer?.cancel();
    int elapsed = 0;
    
    setState(() {
      _isProcessing = true;
      _processingIsCamera = isCamera;
      _currentStepIndex = isCamera ? 1 : 0;
    });

    _processingTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      elapsed += 1;
      if (!mounted || !_isProcessing) {
        timer.cancel();
        return;
      }
      
      setState(() {
        if (elapsed == 3) {
          _currentStepIndex = 1;
        } else if (elapsed == 90) {
          _currentStepIndex = 2;
        } else if (elapsed == 210) {
          _currentStepIndex = 3;
        }
      });
    });
  }

  void _stopProcessingProgress() {
    _processingTimer?.cancel();
    if (mounted) {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);
    _checkServer();
  }

  Future<void> _checkServer() async {
    if (mounted) setState(() => _checkingServer = true);
    final online = await ApiService.healthCheck();
    if (mounted) {
      setState(() {
        _serverOnline = online;
        _checkingServer = false;
      });
    }
  }

  Future<void> _switchToCameraMode() async {
    setState(() {
      _currentMode = RecordMode.camera;
    });
    await _initCamera();
  }

  Future<void> _switchToSelectMode() async {
    _timer?.cancel();
    if (_isRecording) {
      try {
        await _cameraController?.stopVideoRecording();
      } catch (_) {}
    }
    await _cameraController?.dispose();
    _cameraController = null;
    if (mounted) {
      setState(() {
        _isInitialized = false;
        _isRecording = false;
        _currentMode = RecordMode.select;
      });
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        _showSnack('لا توجد كاميرا متاحة في هذا الجهاز');
        _switchToSelectMode();
        return;
      }

      // Prefer front camera for sign language
      final cam = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      _cameraController = CameraController(
        cam,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await _cameraController!.initialize();
      if (mounted) {
        setState(() => _isInitialized = true);
      }
    } catch (e) {
      _showSnack('خطأ في تهيئة الكاميرا: $e');
      _switchToSelectMode();
    }
  }

  void _startRecording() async {
    if (_cameraController == null || !_isInitialized || _isRecording) return;
    if (!_serverOnline) {
      _showSnack('يرجى التأكد من اتصال السيرفر أولاً لبدء الترجمة');
      return;
    }

    try {
      await _cameraController!.startVideoRecording();
      setState(() {
        _isRecording = true;
        _recordingSeconds = 0;
      });

      _timer = Timer.periodic(const Duration(seconds: 1), (t) {
        if (mounted) {
          setState(() {
            _recordingSeconds++;
          });
        }

        // Auto-stop at 10 seconds
        if (_recordingSeconds >= 10) {
          _stopRecording();
        }
      });
    } catch (e) {
      _showSnack('خطأ أثناء بدء التسجيل: $e');
    }
  }

  Future<void> _stopRecording() async {
    if (!_isRecording || _cameraController == null) return;

    _timer?.cancel();
    _startProcessingProgress(true);

    try {
      final videoFile = await _cameraController!.stopVideoRecording();

      // Switch camera mode off and dispose camera controller immediately!
      await _cameraController?.dispose();
      _cameraController = null;
      if (mounted) {
        setState(() {
          _isInitialized = false;
          _currentMode = RecordMode.select;
        });
      }

      // Read video bytes
      final bytes = await videoFile.readAsBytes();

      // Send to server for real video prediction
      final response = await ApiService.predictVideo(
        bytes: bytes,
        path: kIsWeb ? null : videoFile.path,
        fileName: 'camera_capture.mp4',
      );

      // Save to Supabase
      await PredictionService.savePrediction(response);

      _stopProcessingProgress();

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => ResultScreen(
              response: response,
              videoPath: kIsWeb ? '' : videoFile.path,
            ),
          ),
        );
      }
    } catch (e) {
      _stopProcessingProgress();
      if (mounted) {
        _showSnack('خطأ في معالجة الفيديو: $e');
      }
    }
  }

  Future<void> _uploadAndPredictVideo() async {
    if (!_serverOnline) {
      _showSnack('يرجى التأكد من اتصال السيرفر أولاً لرفع الملف');
      return;
    }

    try {
      final result = await FilePicker.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['mp4', 'avi', 'mov', 'mkv', 'webm', 'gif'],
        withData: true,
      );

      if (result == null || result.files.isEmpty) return;

      final file = result.files.first;

      _startProcessingProgress(false);

      // Send file to API
      final response = await ApiService.predictVideo(
        bytes: file.bytes,
        path: kIsWeb ? null : file.path,
        fileName: file.name,
      );

      // Save prediction to Supabase
      await PredictionService.savePrediction(response);

      _stopProcessingProgress();

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => ResultScreen(
              response: response,
              videoPath: kIsWeb ? '' : (file.path ?? ''),
            ),
          ),
        );
      }
    } catch (e) {
      _stopProcessingProgress();
      if (mounted) {
        _showSnack('خطأ في معالجة الملف المرفوع: $e');
      }
    }
  }

  void _showSnack(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg, textDirection: TextDirection.rtl),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }

  @override
  void dispose() {
    _timer?.cancel();
    _processingTimer?.cancel();
    _cameraController?.dispose();
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Render based on active mode
          if (_currentMode == RecordMode.camera)
            _buildCameraView()
          else
            _buildSelectView(),

          // Processing Overlay (Global Stepper)
          if (_isProcessing)
            Positioned.fill(
              child: Container(
                color: Colors.black.withValues(alpha: 0.92),
                child: Center(
                  child: SingleChildScrollView(
                     padding: const EdgeInsets.symmetric(horizontal: 24),
                     child: Container(
                       padding: const EdgeInsets.all(28),
                       decoration: BoxDecoration(
                         color: const Color(0xFF131338),
                         borderRadius: BorderRadius.circular(28),
                         border: Border.all(
                           color: const Color(0xFF6C63FF).withValues(alpha: 0.3),
                           width: 1.5,
                         ),
                         boxShadow: [
                           BoxShadow(
                             color: const Color(0xFF6C63FF).withValues(alpha: 0.15),
                             blurRadius: 30,
                             spreadRadius: 2,
                           ),
                         ],
                       ),
                       child: Column(
                         mainAxisSize: MainAxisSize.min,
                         children: [
                           // Header Glow Icon
                           Container(
                             padding: const EdgeInsets.all(16),
                             decoration: BoxDecoration(
                               color: const Color(0xFF6C63FF).withValues(alpha: 0.1),
                               shape: BoxShape.circle,
                             ),
                             child: const Icon(
                               Icons.psychology_outlined,
                               color: Color(0xFF6C63FF),
                               size: 40,
                             ),
                           ),
                           const SizedBox(height: 20),
                           const Text(
                             'تحليل الإشارة الذكي',
                             style: TextStyle(
                               color: Colors.white,
                               fontSize: 22,
                               fontWeight: FontWeight.bold,
                               fontFamily: 'Inter',
                             ),
                           ),
                           const SizedBox(height: 8),
                           Text(
                             'يرجى إبقاء التطبيق مفتوحاً للترجمة',
                             style: TextStyle(
                               color: Colors.white.withValues(alpha: 0.5),
                               fontSize: 13,
                             ),
                           ),
                           const SizedBox(height: 32),
                           
                           // Timeline Step Rows (Point-to-Point)
                           _buildStepRow(0, 'رفع ملف الفيديو إلى السيرفر', _processingIsCamera),
                           _buildStepConnector(0, _processingIsCamera),
                           _buildStepRow(1, 'استخلاص مفاصل ومعالم الحركة (MediaPipe)', _processingIsCamera),
                           _buildStepConnector(1, _processingIsCamera),
                           _buildStepRow(2, 'توقع وتصنيف الإشارات (SignBart)', _processingIsCamera),
                           _buildStepConnector(2, _processingIsCamera),
                           _buildStepRow(3, 'صياغة النص العربي والترجمة النهائية', _processingIsCamera),
                         ],
                       ),
                     ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  // 1. SELECT VIEW: Beautiful Choice Screen for Camera or Upload Video
  Widget _buildSelectView() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Color(0xFF0A0E21),
            Color(0xFF131338),
            Color(0xFF221554),
          ],
        ),
      ),
      child: SafeArea(
        child: Column(
          children: [
            // Top Bar
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back_ios, color: Colors.white),
                    onPressed: () => Navigator.pop(context),
                  ),
                  const Spacer(),
                  const Text(
                    'ترجمة لغة الإشارة',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const Spacer(),
                  const SizedBox(width: 48), // balance
                ],
              ),
            ),

            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                child: Column(
                  children: [
                    const SizedBox(height: 16),
                    // Server status indicator
                    _buildServerStatusWidget(),
                    const SizedBox(height: 48),

                    // Title instruction
                    const Text(
                      'اختر طريقة ترجمة الإشارة',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'يرجى اختيار التسجيل بالكاميرا أو اختيار فيديو من جهازك',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.5),
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 40),

                    // Option 1: Live Camera Card
                    _buildSelectionCard(
                      title: 'ترجمة عبر الكاميرا',
                      subtitle: 'سجل إشارة حية بالكاميرا وترجمها فوراً',
                      icon: Icons.videocam_rounded,
                      gradient: const [Color(0xFF6C63FF), Color(0xFF4B39EF)],
                      onTap: _switchToCameraMode,
                    ),

                    const SizedBox(height: 24),

                    // Option 2: Upload Video Card
                    _buildSelectionCard(
                      title: 'رفع فيديو أو GIF من الجهاز',
                      subtitle: 'اختر فيديو أو ملف GIF مسجل مسبقاً لترجمته',
                      icon: Icons.upload_file_rounded,
                      gradient: const [Color(0xFFE040FB), Color(0xFF8E24AA)],
                      onTap: _uploadAndPredictVideo,
                    ),
                    
                    const SizedBox(height: 48),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Helper widget for Selection Card
  Widget _buildSelectionCard({
    required String title,
    required String subtitle,
    required IconData icon,
    required List<Color> gradient,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: gradient,
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(24),
          boxShadow: [
            BoxShadow(
              color: gradient[0].withValues(alpha: 0.3),
              blurRadius: 15,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 19,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    subtitle,
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.8),
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 16),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.16),
                shape: BoxShape.circle,
              ),
              child: Icon(
                icon,
                color: Colors.white,
                size: 32,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // 2. CAMERA VIEW: Active Camera Mode
  Widget _buildCameraView() {
    return Stack(
      children: [
        // Camera Preview
        if (_isInitialized && _cameraController != null)
          Positioned.fill(
            child: ClipRRect(
              child: CameraPreview(_cameraController!),
            ),
          )
        else
          Positioned.fill(
            child: Container(
              color: const Color(0xFF0A0E21),
              child: const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(color: Color(0xFF6C63FF)),
                    SizedBox(height: 16),
                    Text(
                      'جاري تشغيل الكاميرا الأمامية...',
                      style: TextStyle(color: Colors.white70, fontSize: 15),
                    ),
                  ],
                ),
              ),
            ),
          ),

        // HUD overlay (Back, recording timer)
        Positioned(
          top: 0,
          left: 0,
          right: 0,
          child: Container(
            padding: EdgeInsets.only(
              top: MediaQuery.of(context).padding.top + 8,
              left: 16,
              right: 16,
              bottom: 20,
            ),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  Colors.black.withValues(alpha: 0.8),
                  Colors.transparent,
                ],
              ),
            ),
            child: Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.arrow_back_ios, color: Colors.white),
                  onPressed: _switchToSelectMode,
                ),
                const Spacer(),
                const Text(
                  'تسجيل الإشارة المباشرة',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const Spacer(),
                const SizedBox(width: 48), // balance
              ],
            ),
          ),
        ),

        // Recording Time Indicator
        if (_isRecording)
          Positioned(
            top: MediaQuery.of(context).padding.top + 80,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.red.withValues(alpha: 0.85),
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.red.withValues(alpha: 0.4),
                      blurRadius: 15,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.fiber_manual_record, color: Colors.white, size: 14),
                    const SizedBox(width: 8),
                    Text(
                      'جاري التسجيل: ${_recordingSeconds}s / 10s',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

        // Bottom control area
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: Container(
            padding: EdgeInsets.only(
              bottom: MediaQuery.of(context).padding.bottom + 28,
              top: 32,
            ),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.bottomCenter,
                end: Alignment.topCenter,
                colors: [
                  Colors.black.withValues(alpha: 0.85),
                  Colors.transparent,
                ],
              ),
            ),
            child: Column(
              children: [
                Text(
                  _isRecording ? 'اضغط لإيقاف التسجيل وبدء الترجمة' : 'اضغط للبدء (التسجيل حتى 10 ثوانٍ)',
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.9),
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 18),
                
                // Camera Action Button
                GestureDetector(
                  onTap: _isRecording ? _stopRecording : _startRecording,
                  child: AnimatedBuilder(
                    animation: _pulseController,
                    builder: (context, child) {
                      final scale = _isRecording
                          ? 1.0 + (_pulseController.value * 0.08)
                          : 1.0;
                      return Transform.scale(
                        scale: scale,
                        child: Container(
                          width: 84,
                          height: 84,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: Colors.white,
                              width: 4,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: (_isRecording ? Colors.red : const Color(0xFF6C63FF))
                                    .withValues(alpha: 0.4),
                                blurRadius: 15,
                                spreadRadius: 2,
                              ),
                            ],
                          ),
                          child: Center(
                            child: AnimatedContainer(
                              duration: const Duration(milliseconds: 200),
                              width: _isRecording ? 32 : 62,
                              height: _isRecording ? 32 : 62,
                              decoration: BoxDecoration(
                                color: _isRecording ? Colors.red : const Color(0xFF6C63FF),
                                borderRadius: BorderRadius.circular(_isRecording ? 8 : 31),
                              ),
                              child: Icon(
                                _isRecording ? Icons.stop_rounded : Icons.videocam_rounded,
                                color: Colors.white,
                                size: 30,
                              ),
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  // Server status pill helper
  Widget _buildServerStatusWidget() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: (_serverOnline
                ? const Color(0xFF00E676)
                : const Color(0xFFFF5252))
            .withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: (_serverOnline
                  ? const Color(0xFF00E676)
                  : const Color(0xFFFF5252))
              .withValues(alpha: 0.3),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_checkingServer)
            const SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Colors.white54,
              ),
            )
          else
            Icon(
              _serverOnline ? Icons.check_circle_outline : Icons.error_outline_rounded,
              color: _serverOnline ? const Color(0xFF00E676) : const Color(0xFFFF5252),
              size: 18,
            ),
          const SizedBox(width: 8),
          Text(
            _checkingServer
                ? 'جاري فحص اتصال السيرفر...'
                : _serverOnline
                    ? 'السيرفر متصل ونشط ✓'
                    : 'السيرفر منفصل وغير نشط',
            style: TextStyle(
              color: _checkingServer
                  ? Colors.white54
                  : _serverOnline
                      ? const Color(0xFF00E676)
                      : const Color(0xFFFF5252),
              fontWeight: FontWeight.bold,
              fontSize: 13,
            ),
          ),
          if (!_checkingServer && !_serverOnline) ...[
            const SizedBox(width: 10),
            GestureDetector(
              onTap: _checkServer,
              child: const Icon(
                Icons.refresh_rounded,
                color: Colors.white,
                size: 16,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildStepRow(int index, String title, bool isCamera) {
    bool isCompleted = index < _currentStepIndex || (isCamera && index == 0);
    bool isActive = index == _currentStepIndex && !(isCamera && index == 0);
    bool isPending = !isCompleted && !isActive;

    Color statusColor;
    String statusText;
    Widget leadingWidget;

    if (isCompleted) {
      statusColor = const Color(0xFF00E676);
      statusText = 'مكتمل ✓';
      leadingWidget = Container(
        width: 32,
        height: 32,
        decoration: const BoxDecoration(
          color: Color(0xFF00E676),
          shape: BoxShape.circle,
        ),
        child: const Icon(Icons.check, color: Colors.black, size: 18),
      );
    } else if (isActive) {
      statusColor = const Color(0xFF6C63FF);
      statusText = 'جاري المعالجة...';
      leadingWidget = SizedBox(
        width: 32,
        height: 32,
        child: Stack(
          alignment: Alignment.center,
          children: [
            Container(
              width: 24,
              height: 24,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFF6C63FF).withValues(alpha: 0.2),
                border: Border.all(color: const Color(0xFF6C63FF), width: 2),
              ),
            ),
            const SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Color(0xFF6C63FF),
              ),
            ),
          ],
        ),
      );
    } else {
      statusColor = Colors.white24;
      statusText = 'في الانتظار';
      leadingWidget = Container(
        width: 32,
        height: 32,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white24, width: 2),
        ),
        child: Center(
          child: Container(
            width: 8,
            height: 8,
            decoration: const BoxDecoration(
              color: Colors.white24,
              shape: BoxShape.circle,
            ),
          ),
        ),
      );
    }

    return Directionality(
      textDirection: TextDirection.rtl,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 4.0),
        child: Row(
          children: [
            leadingWidget,
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      color: isPending ? Colors.white30 : Colors.white,
                      fontSize: 15,
                      fontWeight: isActive ? FontWeight.bold : FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    statusText,
                    style: TextStyle(
                      color: statusColor,
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStepConnector(int index, bool isCamera) {
    bool isCompleted = index < _currentStepIndex || (isCamera && index == 0);
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Row(
        children: [
          Container(
            margin: const EdgeInsets.only(left: 15, right: 15),
            width: 2,
            height: 20,
            color: isCompleted ? const Color(0xFF00E676) : Colors.white10,
          ),
          const Expanded(child: SizedBox()),
        ],
      ),
    );
  }
}
