import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:camera/camera.dart';
import '../services/api_service.dart';
import '../services/reverse_translation_service.dart';

class ChatMessageModel {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final String? translatedFromVideo; // name of video if translated
  String? generatedGifUrl; // if translated to sign GIF
  bool isTranslatingToSign;
  String? translationError;

  ChatMessageModel({
    required this.text,
    required this.isUser,
    required this.timestamp,
    this.translatedFromVideo,
    this.generatedGifUrl,
    this.isTranslatingToSign = false,
    this.translationError,
  });
}

class ChatbotScreen extends StatefulWidget {
  const ChatbotScreen({super.key});

  @override
  State<ChatbotScreen> createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen> with TickerProviderStateMixin {
  final List<ChatMessageModel> _messages = [];
  final ScrollController _scrollController = ScrollController();
  final TextEditingController _textController = TextEditingController();
  
  bool _serverOnline = false;
  bool _checkingServer = true;
  bool _isBotTyping = false;

  // Camera Recording State
  CameraController? _cameraController;
  bool _isCameraInitialized = false;
  bool _isRecording = false;
  bool _showCameraOverlay = false;
  int _recordingSeconds = 0;
  Timer? _recordingTimer;
  late AnimationController _pulseController;

  // Processing Stepper State
  bool _isProcessingVideo = false;
  int _currentStepIndex = 0;
  Timer? _processingTimer;
  String _processingFilename = '';

  // Templates
  final List<Map<String, String>> _templates = [
    {
      'title': 'أهلاً وسهلاً يا صديقي',
      'filename': 'translation_1782029012047.mp4',
      'meaning': 'أهلاً وسهلاً يا صديقي',
    },
    {
      'title': 'أنا مريض',
      'filename': 'translation_1782029110646.mp4',
      'meaning': 'أنا مريض',
    },
    {
      'title': 'أبي يساعد أمي',
      'filename': 'translation_1782029156143.mp4',
      'meaning': 'أبي يساعد أمي',
    },
    {
      'title': 'أنا أحب أخي',
      'filename': 'translation_1782029220619.mp4',
      'meaning': 'أنا أحب أخي',
    },
    {
      'title': 'أنا مريض زكام',
      'filename': 'translation_1782825707280.mp4',
      'meaning': 'أنا مريض زكام',
    },
  ];

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);
    _checkServer();
    // Welcome message
    _messages.add(ChatMessageModel(
      text: 'أهلاً بك! أنا مساعد لغة الإشارة الذكي. يمكنك التحدث معي بإرسال إشارات فيديو أو اختيار نماذج إشارية جاهزة، وسأجيبك بالعربية مع إمكانية تحويل الرد إلى لغة الإشارة.',
      isUser: false,
      timestamp: DateTime.now(),
    ));
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _textController.dispose();
    _recordingTimer?.cancel();
    _processingTimer?.cancel();
    _cameraController?.dispose();
    _pulseController.dispose();
    super.dispose();
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

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  // --- Processing Video overlay simulation ---
  void _startProcessingProgress(String filename) {
    _processingTimer?.cancel();
    _processingFilename = filename;
    int elapsed = 0;
    
    setState(() {
      _isProcessingVideo = true;
      _currentStepIndex = 0;
    });

    _processingTimer = Timer.periodic(const Duration(milliseconds: 1500), (timer) {
      elapsed += 1;
      if (!mounted || !_isProcessingVideo) {
        timer.cancel();
        return;
      }
      
      setState(() {
        if (elapsed == 1) {
          _currentStepIndex = 1; // MediaPipe
        } else if (elapsed == 3) {
          _currentStepIndex = 2; // SignBart
        } else if (elapsed == 4) {
          _currentStepIndex = 3; // Translation text
        }
      });
    });
  }

  void _stopProcessingProgress() {
    _processingTimer?.cancel();
    if (mounted) {
      setState(() {
        _isProcessingVideo = false;
      });
    }
  }

  // --- Video Selection / Upload Flow ---
  Future<void> _uploadVideoSign() async {
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
      _startProcessingProgress(file.name);

      // Call API
      final response = await ApiService.predictVideo(
        bytes: file.bytes,
        path: kIsWeb ? null : file.path,
        fileName: file.name,
        chatbot: true, // Speeds up the fake delay to seconds
      );

      _stopProcessingProgress();
      _handleUserVideoTranslationResult(response.topPrediction.meaning, 'فيديو مرفوع');
    } catch (e) {
      _stopProcessingProgress();
      _showSnack('خطأ في معالجة الملف: $e');
    }
  }

  // --- Template Clicked Flow ---
  Future<void> _sendTemplateSign(Map<String, String> template) async {
    if (!_serverOnline) {
      _showSnack('يرجى التأكد من اتصال السيرفر أولاً');
      return;
    }

    final filename = template['filename']!;
    final title = template['title']!;

    _startProcessingProgress(title);

    try {
      // Send a dummy 1-byte file but with the template filename to trigger mock mapping
      final dummyBytes = Uint8List.fromList([0]);
      
      final response = await ApiService.predictVideo(
        bytes: dummyBytes,
        fileName: filename,
        chatbot: true, // Speeds up fake delay to seconds
      );

      _stopProcessingProgress();
      _handleUserVideoTranslationResult(response.topPrediction.meaning, title);
    } catch (e) {
      _stopProcessingProgress();
      _showSnack('خطأ في معالجة النموذج: $e');
    }
  }

  // --- Camera Recording Flow ---
  Future<void> _startCameraRecording() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        _showSnack('لا توجد كاميرا متاحة');
        return;
      }
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
        setState(() {
          _isCameraInitialized = true;
          _showCameraOverlay = true;
          _isRecording = false;
          _recordingSeconds = 0;
        });
      }
    } catch (e) {
      _showSnack('خطأ في تشغيل الكاميرا: $e');
    }
  }

  void _recordVideo() async {
    if (_cameraController == null || !_isCameraInitialized || _isRecording) return;
    try {
      await _cameraController!.startVideoRecording();
      setState(() {
        _isRecording = true;
        _recordingSeconds = 0;
      });

      _recordingTimer = Timer.periodic(const Duration(seconds: 1), (t) {
        if (mounted) {
          setState(() {
            _recordingSeconds++;
          });
        }
        if (_recordingSeconds >= 10) {
          _stopCameraRecording();
        }
      });
    } catch (e) {
      _showSnack('خطأ أثناء بدء التسجيل: $e');
    }
  }

  Future<void> _stopCameraRecording() async {
    if (!_isRecording || _cameraController == null) return;
    _recordingTimer?.cancel();

    _startProcessingProgress('إشارة الكاميرا الحية');
    setState(() {
      _showCameraOverlay = false;
    });

    try {
      final videoFile = await _cameraController!.stopVideoRecording();
      await _cameraController?.dispose();
      _cameraController = null;
      setState(() {
        _isCameraInitialized = false;
      });

      final bytes = await videoFile.readAsBytes();

      // Send to server
      final response = await ApiService.predictVideo(
        bytes: bytes,
        path: kIsWeb ? null : videoFile.path,
        fileName: 'camera_capture.mp4',
        chatbot: true,
      );

      _stopProcessingProgress();
      _handleUserVideoTranslationResult(response.topPrediction.meaning, 'إشارة كاميرا');
    } catch (e) {
      _stopProcessingProgress();
      _showSnack('خطأ في معالجة فيديو الكاميرا: $e');
    }
  }

  void _cancelCameraOverlay() async {
    _recordingTimer?.cancel();
    if (_isRecording) {
      try {
        await _cameraController?.stopVideoRecording();
      } catch (_) {}
    }
    await _cameraController?.dispose();
    _cameraController = null;
    setState(() {
      _isCameraInitialized = false;
      _showCameraOverlay = false;
      _isRecording = false;
    });
  }

  // --- Processing user text / translation & Chat logic ---
  void _handleUserVideoTranslationResult(String meaning, String videoLabel) {
    if (meaning.isEmpty) {
      _showSnack('فشل السيرفر في التعرف على إشارة الفيديو.');
      return;
    }
    _sendMessage(meaning, videoLabel: videoLabel);
  }

  void _sendTextMessage() {
    final text = _textController.text.trim();
    if (text.isEmpty) return;
    _textController.clear();
    _sendMessage(text);
  }

  Future<void> _sendMessage(String text, {String? videoLabel}) async {
    final userMsg = ChatMessageModel(
      text: text,
      isUser: true,
      timestamp: DateTime.now(),
      translatedFromVideo: videoLabel,
    );

    setState(() {
      _messages.add(userMsg);
      _isBotTyping = true;
    });
    _scrollToBottom();

    // Prepare history
    final List<Map<String, String>> history = _messages
        .where((m) => m != userMsg) // don't include current message in history yet
        .map((m) => {
              'role': m.isUser ? 'user' : 'assistant',
              'content': m.text,
            })
        .toList();

    try {
      final reply = await ApiService.sendChatMessage(text, history);
      if (mounted) {
        setState(() {
          _isBotTyping = false;
          _messages.add(ChatMessageModel(
            text: reply,
            isUser: false,
            timestamp: DateTime.now(),
          ));
        });
        _scrollToBottom();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isBotTyping = false;
          _messages.add(ChatMessageModel(
            text: 'عذراً، حدث خطأ أثناء الاتصال بمساعد المحادثة الذكي. يرجى التحقق من اتصال الإنترنت والسيرفر.',
            isUser: false,
            timestamp: DateTime.now(),
          ));
        });
        _scrollToBottom();
      }
    }
  }

  // --- Translate bot response to Sign GIF ---
  Future<void> _translateResponseToSign(ChatMessageModel message) async {
    setState(() {
      message.isTranslatingToSign = true;
      message.translationError = null;
    });

    try {
      final words = await ReverseTranslationService.analyzeSentence(message.text);
      final gifUrl = await ReverseTranslationService.generateSentenceGif(words);
      setState(() {
        message.generatedGifUrl = gifUrl;
        message.isTranslatingToSign = false;
      });
    } catch (e) {
      setState(() {
        message.isTranslatingToSign = false;
        message.translationError = 'فشل توليد حركة الإشارة. قد يكون خادم الترجمة معطلاً.';
      });
    }
  }

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg, textDirection: TextDirection.rtl),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }

  // --- UI Builders ---
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: SafeArea(
        child: Stack(
          children: [
            Column(
              children: [
                // Header bar
                _buildHeader(),

                // Chat Messages List
                Expanded(
                  child: ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                    itemCount: _messages.length + (_isBotTyping ? 1 : 0),
                    itemBuilder: (context, idx) {
                      if (idx == _messages.length) {
                        return _buildBotTypingBubble();
                      }
                      return _buildChatBubble(_messages[idx]);
                    },
                  ),
                ),

                // Quick Templates Row
                _buildTemplatesRow(),

                // Input Box Bar
                _buildInputBar(),
              ],
            ),

            // Stepper progress indicator overlay
            if (_isProcessingVideo) _buildStepperOverlay(),

            // Camera Overlay
            if (_showCameraOverlay) _buildCameraOverlay(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      decoration: BoxDecoration(
        color: const Color(0xFF0A0E21).withValues(alpha: 0.8),
        border: Border(
          bottom: BorderSide(
            color: Colors.white.withValues(alpha: 0.08),
          ),
        ),
      ),
      child: Row(
        textDirection: TextDirection.rtl,
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFF6C63FF).withValues(alpha: 0.15),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.forum_rounded,
              color: Color(0xFF6C63FF),
              size: 24,
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                const Text(
                  'مساعد الإشارة الذكي (شات بوت)',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 3),
                Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: _serverOnline ? const Color(0xFF00E676) : const Color(0xFFFF5252),
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      _checkingServer
                          ? 'جاري فحص السيرفر...'
                          : _serverOnline
                              ? 'متصل ونشط'
                              : 'غير متصل',
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.5),
                        fontSize: 11,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChatBubble(ChatMessageModel message) {
    final alignRight = message.isUser;
    final bubbleColor = alignRight
        ? const Color(0xFF6C63FF)
        : const Color(0xFF1F1F46);

    return Align(
      alignment: alignRight ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 16),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.8,
        ),
        child: Column(
          crossAxisAlignment: alignRight ? CrossAxisAlignment.end : CrossAxisAlignment.start,
          children: [
            // Label if translated from video
            if (message.translatedFromVideo != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 4, right: 6),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  textDirection: TextDirection.rtl,
                  children: [
                    const Icon(Icons.videocam_rounded, color: Color(0xFF00BFA5), size: 13),
                    const SizedBox(width: 4),
                    Text(
                      'مترجم من: ${message.translatedFromVideo}',
                      style: const TextStyle(color: Color(0xFF00BFA5), fontSize: 11, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
              ),

            // Message Body Container
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: bubbleColor,
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(20),
                  topRight: const Radius.circular(20),
                  bottomLeft: Radius.circular(alignRight ? 20 : 4),
                  bottomRight: Radius.circular(alignRight ? 4 : 20),
                ),
              ),
              child: Text(
                message.text,
                textDirection: TextDirection.rtl,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  height: 1.4,
                ),
              ),
            ),

            // Bot translation to Sign actions
            if (!alignRight) ...[
              const SizedBox(height: 6),
              if (message.generatedGifUrl == null && !message.isTranslatingToSign)
                TextButton.icon(
                  onPressed: () => _translateResponseToSign(message),
                  icon: const Icon(Icons.sign_language_rounded, color: Color(0xFF00BFA5), size: 16),
                  label: const Text(
                    'ترجمة الرد إلى لغة إشارة',
                    style: TextStyle(color: Color(0xFF00BFA5), fontSize: 12, fontWeight: FontWeight.bold),
                  ),
                  style: TextButton.styleFrom(
                    backgroundColor: const Color(0xFF00BFA5).withValues(alpha: 0.08),
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    minimumSize: Size.zero,
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              if (message.isTranslatingToSign)
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF00BFA5)),
                      ),
                      SizedBox(width: 8),
                      Text(
                        'جاري توليد مقاطع الإشارة...',
                        style: TextStyle(color: Color(0xFF00BFA5), fontSize: 11),
                      ),
                    ],
                  ),
                ),
              if (message.translationError != null)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 6),
                  child: Text(
                    message.translationError!,
                    style: const TextStyle(color: Colors.redAccent, fontSize: 11),
                  ),
                ),
              if (message.generatedGifUrl != null) ...[
                const SizedBox(height: 8),
                Container(
                  width: 180,
                  height: 140,
                  decoration: BoxDecoration(
                    color: Colors.black26,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: const Color(0xFF00BFA5).withValues(alpha: 0.3)),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: Image.network(
                    message.generatedGifUrl!,
                    headers: const {
                      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                    },
                    fit: BoxFit.contain,
                    loadingBuilder: (context, child, loadingProgress) {
                      if (loadingProgress == null) return child;
                      return const Center(
                        child: CircularProgressIndicator(color: Color(0xFF00BFA5), strokeWidth: 2),
                      );
                    },
                  ),
                ),
              ],
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildBotTypingBubble() {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 16),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: const BoxDecoration(
          color: Color(0xFF1F1F46),
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(20),
            topRight: Radius.circular(20),
            bottomLeft: Radius.circular(4),
            bottomRight: Radius.circular(20),
          ),
        ),
        child: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white60),
            ),
            SizedBox(width: 10),
            Text(
              'المساعد يفكر ويكتب...',
              style: TextStyle(color: Colors.white70, fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTemplatesRow() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 16, vertical: 4),
          child: Text(
            'إشارات سريعة لل chatbot (النماذج):',
            style: TextStyle(color: Colors.white54, fontSize: 12, fontWeight: FontWeight.bold),
          ),
        ),
        SizedBox(
          height: 48,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            reverse: true, // RTL feel
            padding: const EdgeInsets.symmetric(horizontal: 12),
            itemCount: _templates.length,
            itemBuilder: (context, idx) {
              final template = _templates[idx];
              return Padding(
                padding: const EdgeInsets.symmetric(horizontal: 4),
                child: ActionChip(
                  label: Text(
                    template['title']!,
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                  ),
                  backgroundColor: const Color(0xFF1A1A40),
                  side: BorderSide(color: const Color(0xFF6C63FF).withValues(alpha: 0.3)),
                  onPressed: () => _sendTemplateSign(template),
                ),
              );
            },
          ),
        ),
        const SizedBox(height: 8),
      ],
    );
  }

  Widget _buildInputBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF0A0E21),
        border: Border(
          top: BorderSide(
            color: Colors.white.withValues(alpha: 0.08),
          ),
        ),
      ),
      child: Row(
        children: [
          // Send text button
          IconButton(
            icon: const Icon(Icons.send_rounded, color: Color(0xFF6C63FF)),
            onPressed: _sendTextMessage,
          ),

          // Message input field
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.04),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
              ),
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: TextField(
                controller: _textController,
                style: const TextStyle(color: Colors.white, fontSize: 14),
                textDirection: TextDirection.rtl,
                decoration: const InputDecoration(
                  hintText: 'اكتب رسالة أو أرسل إشارة فيديو...',
                  hintStyle: TextStyle(color: Colors.white30, fontSize: 13),
                  border: InputBorder.none,
                  focusedBorder: InputBorder.none,
                  enabledBorder: InputBorder.none,
                  errorBorder: InputBorder.none,
                  disabledBorder: InputBorder.none,
                  contentPadding: EdgeInsets.symmetric(vertical: 10),
                ),
              ),
            ),
          ),

          // Upload Video Button
          IconButton(
            icon: const Icon(Icons.upload_file_rounded, color: Colors.white60),
            onPressed: _uploadVideoSign,
            tooltip: 'رفع فيديو إشارة',
          ),

          // Camera Record Button
          IconButton(
            icon: const Icon(Icons.videocam_rounded, color: Color(0xFF00BFA5)),
            onPressed: _startCameraRecording,
            tooltip: 'تسجيل إشارة بالكاميرا',
          ),
        ],
      ),
    );
  }

  // --- Processing stepper overlay builder ---
  Widget _buildStepperOverlay() {
    return Positioned.fill(
      child: Container(
        color: Colors.black.withValues(alpha: 0.88),
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
                  ),
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
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
                    'تحليل ومعالجة إشارة الفيديو',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'المنفذ: $_processingFilename',
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.4),
                      fontSize: 12,
                    ),
                  ),
                  const SizedBox(height: 28),
                  
                  _buildStepRow(0, 'رفع ملف الفيديو إلى السيرفر', false),
                  _buildStepConnector(0, false),
                  _buildStepRow(1, 'استخلاص مفاصل ومعالم الحركة (MediaPipe)', false),
                  _buildStepConnector(1, false),
                  _buildStepRow(2, 'توقع وتصنيف الإشارات (SignBart)', false),
                  _buildStepConnector(2, false),
                  _buildStepRow(3, 'صياغة النص العربي والترجمة النهائية', false),
                ],
              ),
            ),
          ),
        ),
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
        width: 30,
        height: 30,
        decoration: const BoxDecoration(
          color: Color(0xFF00E676),
          shape: BoxShape.circle,
        ),
        child: const Icon(Icons.check, color: Colors.black, size: 16),
      );
    } else if (isActive) {
      statusColor = const Color(0xFF6C63FF);
      statusText = 'جاري المعالجة...';
      leadingWidget = SizedBox(
        width: 30,
        height: 30,
        child: Stack(
          alignment: Alignment.center,
          children: [
            Container(
              width: 22,
              height: 22,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFF6C63FF).withValues(alpha: 0.2),
                border: Border.all(color: const Color(0xFF6C63FF), width: 1.5),
              ),
            ),
            const SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(
                strokeWidth: 1.5,
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
        width: 30,
        height: 30,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white24, width: 1.5),
        ),
        child: Center(
          child: Container(
            width: 6,
            height: 6,
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
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      color: isPending ? Colors.white30 : Colors.white,
                      fontSize: 14,
                      fontWeight: isActive ? FontWeight.bold : FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    statusText,
                    style: TextStyle(
                      color: statusColor,
                      fontSize: 11,
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
            margin: const EdgeInsets.only(left: 14, right: 14),
            width: 1.5,
            height: 16,
            color: isCompleted ? const Color(0xFF00E676) : Colors.white10,
          ),
          const Expanded(child: SizedBox()),
        ],
      ),
    );
  }

  // --- Camera Overlay Builder ---
  Widget _buildCameraOverlay() {
    return Positioned.fill(
      child: Container(
        color: Colors.black,
        child: Stack(
          children: [
            // Camera preview
            if (_isCameraInitialized && _cameraController != null)
              Positioned.fill(
                child: ClipRRect(
                  child: CameraPreview(_cameraController!),
                ),
              )
            else
              const Center(
                child: CircularProgressIndicator(color: Color(0xFF6C63FF)),
              ),

            // Top control hud
            Positioned(
              top: 20,
              left: 16,
              right: 16,
              child: Row(
                textDirection: TextDirection.rtl,
                children: [
                  IconButton(
                    icon: const Icon(Icons.close_rounded, color: Colors.white, size: 28),
                    onPressed: _cancelCameraOverlay,
                  ),
                  const Spacer(),
                  const Text(
                    'تسجيل إشارة للمحادثة',
                    style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  const Spacer(),
                  const SizedBox(width: 48),
                ],
              ),
            ),

            // Recording Timer Badge
            if (_isRecording)
              Positioned(
                top: 80,
                left: 0,
                right: 0,
                child: Center(
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.red.withValues(alpha: 0.85),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.fiber_manual_record, color: Colors.white, size: 12),
                        const SizedBox(width: 6),
                        Text(
                          'تسجيل: $_recordingSecondsث / 10ث',
                          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                ),
              ),

            // Bottom capture button
            Positioned(
              bottom: 40,
              left: 0,
              right: 0,
              child: Column(
                children: [
                  Text(
                    _isRecording ? 'اضغط لإيقاف التسجيل والإرسال' : 'اضغط لبدء تسجيل إشارتك بالكاميرا',
                    style: const TextStyle(color: Colors.white70, fontSize: 13),
                  ),
                  const SizedBox(height: 16),
                  GestureDetector(
                    onTap: _isRecording ? _stopCameraRecording : _recordVideo,
                    child: AnimatedBuilder(
                      animation: _pulseController,
                      builder: (context, child) {
                        final scale = _isRecording ? 1.0 + (_pulseController.value * 0.08) : 1.0;
                        return Transform.scale(
                          scale: scale,
                          child: Container(
                            width: 80,
                            height: 80,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(color: Colors.white, width: 4),
                            ),
                            child: Center(
                              child: Container(
                                width: _isRecording ? 28 : 58,
                                height: _isRecording ? 28 : 58,
                                decoration: BoxDecoration(
                                  color: _isRecording ? Colors.red : const Color(0xFF6C63FF),
                                  borderRadius: BorderRadius.circular(_isRecording ? 6 : 29),
                                ),
                                child: Icon(
                                  _isRecording ? Icons.stop_rounded : Icons.videocam_rounded,
                                  color: Colors.white,
                                  size: 26,
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
          ],
        ),
      ),
    );
  }
}
