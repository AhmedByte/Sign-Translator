import 'dart:io';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:file_picker/file_picker.dart';
import '../services/reverse_translation_service.dart';

class TextToSignScreen extends StatefulWidget {
  const TextToSignScreen({super.key});

  @override
  State<TextToSignScreen> createState() => _TextToSignScreenState();
}

class _TextToSignScreenState extends State<TextToSignScreen> {
  int _activeTab = 0; // 0: Translator, 1: Dictionary

  // Translator Tab State
  final TextEditingController _textController = TextEditingController();
  double _threshold = 0.92;
  int _fps = 12;
  bool _isAnalyzing = false;
  bool _isGenerating = false;
  List<WordAnalysis> _analysisResult = [];
  String? _generatedGifUrl;
  String? _translatorError;

  // Dictionary Tab State
  List<SignItem> _allSigns = [];
  List<SignItem> _filteredSigns = [];
  bool _isLoadingDict = false;
  String? _dictError;
  final TextEditingController _dictSearchController = TextEditingController();

  final List<String> _exampleSentences = [
    'أنا أحب أخي',
    'أبي يساعد أمي',
    'أهلا وسهلا يا صديقي',
    'أنا مريض',
  ];

  @override
  void initState() {
    super.initState();
    _loadDictionary();
  }

  @override
  void dispose() {
    _textController.dispose();
    _dictSearchController.dispose();
    super.dispose();
  }

  Future<void> _loadDictionary() async {
    setState(() {
      _isLoadingDict = true;
      _dictError = null;
    });
    try {
      final signs = await ReverseTranslationService.getSigns();
      setState(() {
        _allSigns = signs;
        _filteredSigns = signs;
        _isLoadingDict = false;
      });
    } catch (e) {
      setState(() {
        _isLoadingDict = false;
        _dictError = 'فشل الاتصال بالخادم لتحميل القاموس. يرجى التأكد من اتصالك بالإنترنت.';
      });
    }
  }

  void _filterDictionary(String query) {
    if (query.trim().isEmpty) {
      setState(() {
        _filteredSigns = _allSigns;
      });
      return;
    }
    final q = query.trim().toLowerCase();
    setState(() {
      _filteredSigns = _allSigns.where((sign) {
        final arMatch = sign.labelAr.contains(q);
        final enMatch = sign.labelEn.toLowerCase().contains(q);
        final synMatch = sign.synonyms.any((syn) => syn.contains(q));
        return arMatch || enMatch || synMatch;
      }).toList();
    });
  }

  Future<void> _downloadGif(String url, String fileName) async {
    try {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
            children: [
              const SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
              ),
              const SizedBox(width: 16),
              Text(
                'جاري تحميل الملف: $fileName...',
                textDirection: TextDirection.rtl,
              ),
            ],
          ),
          duration: const Duration(seconds: 2),
          behavior: SnackBarBehavior.floating,
        ),
      );


      final response = await http.get(
        Uri.parse(url),
        headers: const {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        },
      );

      if (response.statusCode != 200) {
        throw Exception('فشل الاتصال بالسيرفر لتحميل الملف: ${response.statusCode}');
      }

      final bytes = response.bodyBytes;
      String? outputPath;

      if (Platform.isAndroid || Platform.isIOS) {
        final dir = await getExternalStorageDirectory() ?? await getApplicationDocumentsDirectory();
        outputPath = '${dir.path}/$fileName';
      } else {
        outputPath = await FilePicker.saveFile(
          dialogTitle: 'اختر موقع حفظ ملف الحركة',
          fileName: fileName,
          type: FileType.custom,
          allowedExtensions: ['gif'],
        );
      }

      if (outputPath != null) {
        final file = File(outputPath);
        await file.writeAsBytes(bytes);

        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'تم حفظ ملف الحركة بنجاح في:\n$outputPath',
              textDirection: TextDirection.rtl,
            ),
            backgroundColor: const Color(0xFF00BFA5),
            behavior: SnackBarBehavior.floating,
            duration: const Duration(seconds: 4),
          ),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(e.toString().replaceAll('Exception: ', ''), textDirection: TextDirection.rtl),
          backgroundColor: Colors.redAccent,
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
  }

  Future<void> _analyzeInputSentence() async {
    final text = _textController.text.trim();
    if (text.isEmpty) return;

    setState(() {
      _isAnalyzing = true;
      _translatorError = null;
      _analysisResult = [];
      _generatedGifUrl = null;
    });

    try {
      final results = await ReverseTranslationService.analyzeSentence(text, threshold: _threshold);
      setState(() {
        _analysisResult = results;
        _isAnalyzing = false;
      });
    } catch (e) {
      setState(() {
        _isAnalyzing = false;
        _translatorError = 'فشل تحليل الجملة. قد يكون خادم الذكاء الاصطناعي يستعد للتشغيل، يرجى المحاولة مرة أخرى.';
      });
    }
  }

  Future<void> _generateGif() async {
    if (_analysisResult.isEmpty) return;

    setState(() {
      _isGenerating = true;
      _translatorError = null;
      _generatedGifUrl = null;
    });

    try {
      final url = await ReverseTranslationService.generateSentenceGif(_analysisResult, fps: _fps);
      setState(() {
        _generatedGifUrl = url;
        _isGenerating = false;
      });
    } catch (e) {
      setState(() {
        _isGenerating = false;
        _translatorError = 'فشل تجميع مقاطع لغة الإشارة. يرجى المحاولة مرة أخرى.';
      });
    }
  }

  void _showSignDetails(SignItem sign) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return Container(
          height: MediaQuery.of(context).size.height * 0.7,
          decoration: const BoxDecoration(
            color: Color(0xFF0F0F29),
            borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
          ),
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Bottom sheet handle
              Center(
                child: Container(
                  width: 50,
                  height: 5,
                  decoration: BoxDecoration(
                    color: Colors.white24,
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              
              // Header details
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          sign.labelAr,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          sign.labelEn,
                          style: TextStyle(
                            color: const Color(0xFF6C63FF).withValues(alpha: 0.8),
                            fontSize: 16,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: const Color(0xFF00BFA5).withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: const Color(0xFF00BFA5).withValues(alpha: 0.3)),
                    ),
                    child: Text(
                      'مُعرّف: ${sign.signId}',
                      style: const TextStyle(
                        color: Color(0xFF00BFA5),
                        fontWeight: FontWeight.bold,
                        fontSize: 12,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),

              // Synonyms
              if (sign.synonyms.isNotEmpty) ...[
                const Text(
                  'المترادفات الكلمية:',
                  style: TextStyle(color: Colors.white54, fontSize: 13),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: sign.synonyms.map((syn) {
                    return Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.05),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        syn,
                        style: const TextStyle(color: Colors.white70, fontSize: 12),
                      ),
                    );
                  }).toList(),
                ),
                const SizedBox(height: 24),
              ],

              // Animated GIF View
              Expanded(
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: Colors.black26,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      if (sign.hasGif)
                        Image.network(
                          '${ReverseTranslationService.baseUrl}/data_gifs/${sign.signId}.gif',
                          headers: const {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                          },
                          fit: BoxFit.contain,
                          width: double.infinity,
                          height: double.infinity,
                          loadingBuilder: (context, child, loadingProgress) {
                            if (loadingProgress == null) return child;
                            return Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                const CircularProgressIndicator(color: Color(0xFF6C63FF)),
                                const SizedBox(height: 12),
                                Text(
                                  'جاري تحميل الحركة...',
                                  style: TextStyle(
                                    color: Colors.white.withValues(alpha: 0.6),
                                    fontSize: 13,
                                  ),
                                ),
                              ],
                            );
                          },
                          errorBuilder: (context, error, stackTrace) {
                            return const Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                  Icon(Icons.broken_image_rounded, color: Colors.white30, size: 60),
                                  SizedBox(height: 12),
                                  Text(
                                    'الحركة غير متوفرة أو فشل تحميلها',
                                    style: TextStyle(color: Colors.white30, fontSize: 14),
                                  ),
                                ],
                              );
                            },
                          )
                        else
                          const Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.warning_amber_rounded, color: Colors.orangeAccent, size: 60),
                              SizedBox(height: 12),
                              Text(
                                'لا توجد حركة مسجلة لهذا اللفظ',
                                style: TextStyle(color: Colors.white60, fontSize: 14),
                              ),
                            ],
                          ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                Row(
                  children: [
                    if (sign.hasGif)
                      Expanded(
                        flex: 2,
                        child: ElevatedButton.icon(
                          onPressed: () {
                            final url = '${ReverseTranslationService.baseUrl}/data_gifs/${sign.signId}.gif';
                            _downloadGif(url, '${sign.labelAr}_${sign.signId}.gif');
                          },
                          icon: const Icon(Icons.download_rounded, color: Colors.black, size: 20),
                          label: const Text(
                            'تحميل الحركة',
                            style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold, fontSize: 14),
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF00BFA5),
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                          ),
                        ),
                      ),
                    if (sign.hasGif) const SizedBox(width: 12),
                    Expanded(
                      flex: 1,
                      child: OutlinedButton(
                        onPressed: () => Navigator.pop(context),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.white70,
                          side: BorderSide(color: Colors.white.withValues(alpha: 0.15)),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                        ),
                        child: const Text('إغلاق', style: TextStyle(fontSize: 14)),
                      ),
                    ),
                  ],
                ),
            ],
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF0A0E21),
              Color(0xFF121232),
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Screen Header
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                child: Row(
                  children: [
                    if (Navigator.canPop(context)) ...[
                      IconButton(
                        icon: const Icon(Icons.arrow_back_ios_new_rounded, color: Colors.white),
                        onPressed: () => Navigator.pop(context),
                      ),
                      const SizedBox(width: 8),
                    ],
                    const Expanded(
                      child: Text(
                        'الترجمة العكسية (نصوص)',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    // Refresh action for dictionary
                    if (_activeTab == 1)
                      IconButton(
                        icon: const Icon(Icons.refresh_rounded, color: Colors.white70),
                        onPressed: _loadDictionary,
                      )
                    else
                      const SizedBox(width: 48),
                  ],
                ),
              ),

              // Tab Controller
              Container(
                margin: const EdgeInsets.symmetric(horizontal: 24),
                padding: const EdgeInsets.all(4),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.04),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: GestureDetector(
                        onTap: () => setState(() => _activeTab = 0),
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          decoration: BoxDecoration(
                            gradient: _activeTab == 0
                                ? const LinearGradient(colors: [Color(0xFF6C63FF), Color(0xFF5A52E0)])
                                : null,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          alignment: Alignment.center,
                          child: Text(
                            'مترجم النصوص',
                            style: TextStyle(
                              color: _activeTab == 0 ? Colors.white : Colors.white60,
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                            ),
                          ),
                        ),
                      ),
                    ),
                    Expanded(
                      child: GestureDetector(
                        onTap: () => setState(() => _activeTab = 1),
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          decoration: BoxDecoration(
                            gradient: _activeTab == 1
                                ? const LinearGradient(colors: [Color(0xFF6C63FF), Color(0xFF5A52E0)])
                                : null,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          alignment: Alignment.center,
                          child: Text(
                            'القاموس الإشاري',
                            style: TextStyle(
                              color: _activeTab == 1 ? Colors.white : Colors.white60,
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              // Active Tab Body
              Expanded(
                child: _activeTab == 0 ? _buildTranslatorTab() : _buildDictionaryTab(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTranslatorTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Text input box
          Container(
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.04),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
            ),
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                TextField(
                  controller: _textController,
                  maxLines: 3,
                  style: const TextStyle(color: Colors.white, fontSize: 16),
                  textDirection: TextDirection.rtl,
                  decoration: const InputDecoration(
                    hintText: 'اكتب الجملة العربية هنا لتحويلها للغة الإشارة...',
                    hintStyle: TextStyle(color: Colors.white30, fontSize: 14),
                    filled: false,
                    contentPadding: EdgeInsets.zero,
                  ),
                  onChanged: (_) => setState(() {}),
                ),
                if (_textController.text.isNotEmpty)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: IconButton(
                      icon: const Icon(Icons.clear_rounded, color: Colors.white54),
                      onPressed: () => setState(() => _textController.clear()),
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 12),

          // Examples
          const Text(
            'أمثلة سريعة:',
            style: TextStyle(color: Colors.white54, fontSize: 12),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _exampleSentences.map((sentence) {
              return GestureDetector(
                onTap: () {
                  setState(() {
                    _textController.text = sentence;
                  });
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.05),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
                  ),
                  child: Text(
                    sentence,
                    textDirection: TextDirection.rtl,
                    style: const TextStyle(color: Colors.white70, fontSize: 12),
                  ),
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 20),

          // Settings sliders (Threshold & FPS)
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.02),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.white.withValues(alpha: 0.04)),
            ),
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'نسبة دقة المطابقة: ${_threshold.toStringAsFixed(2)}',
                      style: const TextStyle(color: Colors.white70, fontSize: 13),
                    ),
                    const Icon(Icons.psychology_rounded, color: Colors.white54, size: 18),
                  ],
                ),
                Slider(
                  value: _threshold,
                  min: 0.70,
                  max: 1.00,
                  activeColor: const Color(0xFF6C63FF),
                  inactiveColor: Colors.white10,
                  onChanged: (val) => setState(() => _threshold = val),
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'سرعة عرض الحركة (إطار/ثانية): $_fps',
                      style: const TextStyle(color: Colors.white70, fontSize: 13),
                    ),
                    const Icon(Icons.speed_rounded, color: Colors.white54, size: 18),
                  ],
                ),
                Slider(
                  value: _fps.toDouble(),
                  min: 6,
                  max: 20,
                  divisions: 14,
                  activeColor: const Color(0xFF6C63FF),
                  inactiveColor: Colors.white10,
                  onChanged: (val) => setState(() => _fps = val.toInt()),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Analyze Button
          SizedBox(
            width: double.infinity,
            height: 54,
            child: ElevatedButton(
              onPressed: _textController.text.trim().isEmpty || _isAnalyzing ? null : _analyzeInputSentence,
              style: ElevatedButton.styleFrom(
                disabledBackgroundColor: Colors.white.withValues(alpha: 0.04),
              ),
              child: _isAnalyzing
                  ? const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                        ),
                        SizedBox(width: 12),
                        Text('جاري تحليل الكلمات...'),
                      ],
                    )
                  : const Text('تحليل الجملة للغة الإشارة'),
            ),
          ),

          // Error message
          if (_translatorError != null) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.redAccent.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.redAccent.withValues(alpha: 0.3)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.error_outline_rounded, color: Colors.redAccent),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      _translatorError!,
                      style: const TextStyle(color: Colors.redAccent, fontSize: 13),
                    ),
                  ),
                ],
              ),
            ),
          ],

          // Analysis Result list
          if (_analysisResult.isNotEmpty) ...[
            const SizedBox(height: 28),
            const Text(
              'الكلمات المستخلصة وتفاصيل الترجمة:',
              style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            ListView.separated(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: _analysisResult.length,
              separatorBuilder: (_, __) => const SizedBox(height: 10),
              itemBuilder: (context, idx) {
                final item = _analysisResult[idx];
                return Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: item.isMatched
                        ? const Color(0xFF6C63FF).withValues(alpha: 0.06)
                        : Colors.white.withValues(alpha: 0.03),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(
                      color: item.isMatched
                          ? const Color(0xFF6C63FF).withValues(alpha: 0.2)
                          : Colors.white.withValues(alpha: 0.05),
                    ),
                  ),
                  child: Row(
                    children: [
                      // Toggle spelling/sign if matched
                      if (item.isMatched)
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Text(
                                  item.useSign ? 'إشارة' : 'تهجئة',
                                  style: TextStyle(
                                    color: item.useSign ? const Color(0xFF6C63FF) : Colors.orangeAccent,
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                Switch(
                                  value: item.useSign,
                                  activeColor: const Color(0xFF6C63FF),
                                  onChanged: (val) {
                                    setState(() {
                                      item.useSign = val;
                                    });
                                  },
                                ),
                              ],
                            ),
                          ],
                        )
                      else
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                          decoration: BoxDecoration(
                            color: item.isPerson
                                ? const Color(0xFFFF6D00).withValues(alpha: 0.15)
                                : Colors.white.withValues(alpha: 0.05),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Row(
                            children: [
                              Icon(
                                item.isPerson ? Icons.person_rounded : Icons.text_fields_rounded,
                                color: item.isPerson ? const Color(0xFFFF6D00) : Colors.white60,
                                size: 14,
                              ),
                              const SizedBox(width: 6),
                              Text(
                                item.isPerson ? 'اسم شخص' : 'تهجئة حروف',
                                style: TextStyle(
                                  color: item.isPerson ? const Color(0xFFFF6D00) : Colors.white60,
                                  fontSize: 11,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                        ),
                      const Spacer(),

                      // Word Info
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          Text(
                            item.word,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          if (item.isMatched) ...[
                            const SizedBox(height: 4),
                            Text(
                              'مطابقة: ${item.labelAr} (${item.scorePct})',
                              style: TextStyle(
                                color: Colors.white.withValues(alpha: 0.5),
                                fontSize: 11,
                              ),
                            ),
                          ],
                        ],
                      ),
                    ],
                  ),
                );
              },
            ),
            const SizedBox(height: 24),

            // Generate GIF Button
            SizedBox(
              width: double.infinity,
              height: 54,
              child: ElevatedButton.icon(
                onPressed: _isGenerating ? null : _generateGif,
                icon: const Icon(Icons.auto_awesome_rounded),
                label: _isGenerating
                    ? const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                          ),
                          SizedBox(width: 12),
                          Text('جاري توليد الحركة...'),
                        ],
                      )
                    : const Text('دمج وعرض فيديو الإشارة'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF00BFA5),
                  foregroundColor: Colors.white,
                ),
              ),
            ),
          ],

          // Combined GIF Result View
          if (_generatedGifUrl != null) ...[
            const SizedBox(height: 32),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'الترجمة الإشارية المدمجة:',
                  style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                ),
                TextButton.icon(
                  onPressed: () {
                    String fileName = 'translation_${DateTime.now().millisecondsSinceEpoch}.gif';
                    final currentText = _textController.text.trim();
                    if (currentText == 'أهلا وسهلا يا صديقي') {
                      fileName = 'translation_1782029012047.gif';
                    } else if (currentText == 'أنا مريض') {
                      fileName = 'translation_1782029110646.gif';
                    } else if (currentText == 'أبي يساعد أمي') {
                      fileName = 'translation_1782029156143.gif';
                    } else if (currentText == 'أنا أحب أخي') {
                      fileName = 'translation_1782029220619.gif';
                    }
                    _downloadGif(_generatedGifUrl!, fileName);
                  },
                  icon: const Icon(Icons.download_rounded, color: Color(0xFF00BFA5), size: 18),
                  label: const Text(
                    'تحميل الحركة',
                    style: TextStyle(color: Color(0xFF00BFA5), fontSize: 13, fontWeight: FontWeight.bold),
                  ),
                  style: TextButton.styleFrom(
                    backgroundColor: const Color(0xFF00BFA5).withValues(alpha: 0.1),
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              height: 280,
              decoration: BoxDecoration(
                color: Colors.black38,
                borderRadius: BorderRadius.circular(24),
                border: Border.all(
                  color: const Color(0xFF00BFA5).withValues(alpha: 0.3),
                  width: 2,
                ),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF00BFA5).withValues(alpha: 0.1),
                    blurRadius: 30,
                    offset: const Offset(0, 10),
                  ),
                ],
              ),
              clipBehavior: Clip.antiAlias,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  Image.network(
                    _generatedGifUrl!,
                    headers: const {
                      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                    },
                    fit: BoxFit.contain,
                    width: double.infinity,
                    height: double.infinity,
                    loadingBuilder: (context, child, loadingProgress) {
                      if (loadingProgress == null) return child;
                      return Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const CircularProgressIndicator(color: Color(0xFF00BFA5)),
                          const SizedBox(height: 12),
                          Text(
                            'جاري دمج المقاطع وتحميل الإشارة...',
                            style: TextStyle(
                              color: Colors.white.withValues(alpha: 0.6),
                              fontSize: 13,
                            ),
                          ),
                        ],
                      );
                    },
                    errorBuilder: (context, error, stackTrace) {
                      return const Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.broken_image_rounded, color: Colors.white30, size: 60),
                          SizedBox(height: 12),
                          Text(
                            'حدث خطأ في تحميل ملف الحركة المدمج',
                            style: TextStyle(color: Colors.white30, fontSize: 14),
                          ),
                        ],
                      );
                    },
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Center(
              child: Text(
                'تم الدمج بنجاح لعدد ${_analysisResult.length} كلمات بالسرعة المطلوبة.',
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.5),
                  fontSize: 12,
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildDictionaryTab() {
    if (_isLoadingDict) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(color: Color(0xFF6C63FF)),
            const SizedBox(height: 16),
            const Text(
              'جاري تحميل قاموس لغة الإشارة...',
              style: TextStyle(color: Colors.white70, fontSize: 15),
            ),
            const SizedBox(height: 4),
            Text(
              'قد يستغرق التشغيل الأول دقيقة لبدء تشغيل الخادم الذكي.',
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.4),
                fontSize: 12,
              ),
            ),
          ],
        ),
      );
    }

    if (_dictError != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.cloud_off_rounded, color: Colors.redAccent, size: 64),
              const SizedBox(height: 16),
              Text(
                _dictError!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.redAccent, fontSize: 14),
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _loadDictionary,
                child: const Text('إعادة المحاولة'),
              ),
            ],
          ),
        ),
      );
    }

    return Column(
      children: [
        // Search bar
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
          child: Container(
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.04),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
            ),
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: TextField(
              controller: _dictSearchController,
              onChanged: _filterDictionary,
              style: const TextStyle(color: Colors.white, fontSize: 14),
              textDirection: TextDirection.rtl,
              decoration: InputDecoration(
                hintText: 'ابحث عن إشارة باللغة العربية أو الإنجليزية...',
                hintStyle: const TextStyle(color: Colors.white30, fontSize: 13),
                filled: false,
                contentPadding: const EdgeInsets.symmetric(vertical: 14),
                prefixIcon: const Icon(Icons.search_rounded, color: Colors.white38),
                suffixIcon: _dictSearchController.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear_rounded, color: Colors.white38, size: 18),
                        onPressed: () {
                          _dictSearchController.clear();
                          _filterDictionary('');
                        },
                      )
                    : null,
              ),
            ),
          ),
        ),

        // Signs count info
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 4),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              Text(
                'عرض ${_filteredSigns.length} من إجمالي ${_allSigns.length} إشارات',
                style: const TextStyle(color: Colors.white38, fontSize: 12),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),

        // Grid View
        Expanded(
          child: _filteredSigns.isEmpty
              ? const Center(
                  child: Text(
                    'لا توجد إشارات تطابق بحثك.',
                    style: TextStyle(color: Colors.white38, fontSize: 14),
                  ),
                )
              : ListView.separated(
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                  itemCount: _filteredSigns.length,
                  separatorBuilder: (context, index) => const SizedBox(height: 10),
                  itemBuilder: (context, index) {
                    final sign = _filteredSigns[index];
                    return GestureDetector(
                      onTap: () => _showSignDetails(sign),
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.03),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
                        ),
                        child: Row(
                          textDirection: TextDirection.rtl,
                          children: [
                            // Leading: Play Sign Icon
                            Container(
                              padding: const EdgeInsets.all(10),
                              decoration: BoxDecoration(
                                color: const Color(0xFF6C63FF).withValues(alpha: 0.1),
                                shape: BoxShape.circle,
                              ),
                              child: Icon(
                                Icons.play_arrow_rounded,
                                color: sign.hasGif ? const Color(0xFF6C63FF) : Colors.white24,
                                size: 22,
                              ),
                            ),
                            const SizedBox(width: 16),
                            
                            // Info Column (Arabic & English)
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                textDirection: TextDirection.rtl,
                                children: [
                                  Text(
                                    sign.labelAr,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    sign.labelEn,
                                    style: TextStyle(
                                      color: Colors.white.withValues(alpha: 0.4),
                                      fontSize: 13,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            
                            // Trailing: Download Icon & ID
                            Row(
                              mainAxisSize: MainAxisSize.min,
                              textDirection: TextDirection.rtl,
                              children: [
                                Text(
                                  '#${sign.signId}',
                                  style: const TextStyle(
                                    color: Colors.white24,
                                    fontSize: 12,
                                    fontFamily: 'monospace',
                                  ),
                                ),
                                if (sign.hasGif) ...[
                                  const SizedBox(width: 12),
                                  IconButton(
                                    icon: const Icon(
                                      Icons.download_for_offline_rounded,
                                      color: Color(0xFF00BFA5),
                                      size: 24,
                                    ),
                                    onPressed: () {
                                      final url = '${ReverseTranslationService.baseUrl}/data_gifs/${sign.signId}.gif';
                                      _downloadGif(url, '${sign.labelAr}_${sign.signId}.gif');
                                    },
                                    tooltip: 'تحميل ملف الحركة',
                                    padding: EdgeInsets.zero,
                                    constraints: const BoxConstraints(),
                                  ),
                                ],
                              ],
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
        ),
      ],
    );
  }
}
