import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:share_plus/share_plus.dart';
import '../services/api_service.dart';

class ResultScreen extends StatelessWidget {
  final PredictResponse response;
  final String videoPath;

  const ResultScreen({
    super.key,
    required this.response,
    required this.videoPath,
  });

  @override
  Widget build(BuildContext context) {
    final top = response.topPrediction;

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF0A0E21),
              Color(0xFF1A1A40),
              Color(0xFF2D1B69),
            ],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              children: [
                // Header
                Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back_ios,
                          color: Colors.white),
                      onPressed: () => Navigator.pop(context),
                    ),
                    const Spacer(),
                    const Text(
                      'نتيجة الترجمة',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const Spacer(),
                    const SizedBox(width: 48),
                  ],
                ),
                const SizedBox(height: 32),

                // Main result card
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(28),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        const Color(0xFF6C63FF).withValues(alpha: 0.3),
                        const Color(0xFF6C63FF).withValues(alpha: 0.1),
                      ],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(
                      color: const Color(0xFF6C63FF).withValues(alpha: 0.3),
                    ),
                  ),
                  child: Column(
                    children: [
                      // Confidence badge
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 6),
                        decoration: BoxDecoration(
                          color: _confidenceColor(top.confidence)
                              .withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(
                          'ثقة ${top.confidence.toStringAsFixed(1)}%',
                          style: TextStyle(
                            color: _confidenceColor(top.confidence),
                            fontWeight: FontWeight.bold,
                            fontSize: 14,
                          ),
                        ),
                      ),
                      const SizedBox(height: 20),

                      // Gloss
                      const Text(
                        'الإشارة',
                        style: TextStyle(
                          color: Colors.white54,
                          fontSize: 13,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        top.gloss,
                        textAlign: TextAlign.center,
                        textDirection: TextDirection.rtl,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 26,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 20),

                      // Divider
                      Container(
                        height: 1,
                        color: Colors.white.withValues(alpha: 0.1),
                      ),
                      const SizedBox(height: 20),

                      // Meaning
                      const Text(
                        'المعنى',
                        style: TextStyle(
                          color: Colors.white54,
                          fontSize: 13,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        top.meaning.isEmpty ? '—' : top.meaning,
                        textAlign: TextAlign.center,
                        textDirection: TextDirection.rtl,
                        style: const TextStyle(
                          color: Color(0xFF6C63FF),
                          fontSize: 20,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 16),
                      // Actions Row
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          IconButton(
                            icon: const Icon(Icons.share_rounded, color: Colors.white70),
                            onPressed: () => Share.share(top.meaning.isEmpty ? top.gloss : top.meaning),
                            tooltip: 'مشاركة',
                          ),
                          const SizedBox(width: 8),
                          IconButton(
                            icon: const Icon(Icons.search_rounded, color: Colors.white70),
                            onPressed: () => launchUrl(Uri.parse('https://www.google.com/search?q=${Uri.encodeComponent(top.meaning.isEmpty ? top.gloss : top.meaning)}')),
                            tooltip: 'بحث',
                          ),
                          const SizedBox(width: 8),
                          IconButton(
                            icon: const Icon(Icons.chat_rounded, color: Color(0xFF25D366)), // WhatsApp Color
                            onPressed: () => launchUrl(Uri.parse('https://wa.me/?text=${Uri.encodeComponent(top.meaning.isEmpty ? top.gloss : top.meaning)}')),
                            tooltip: 'واتساب',
                          ),
                          const SizedBox(width: 8),
                          IconButton(
                            icon: const Icon(Icons.auto_awesome_rounded, color: Colors.purpleAccent),
                            onPressed: () => launchUrl(Uri.parse('https://chatgpt.com/?q=${Uri.encodeComponent(top.meaning.isEmpty ? top.gloss : top.meaning)}')),
                            tooltip: 'شات جي بي تي',
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                // Top 5
                Align(
                  alignment: Alignment.centerRight,
                  child: Text(
                    'أعلى 5 نتائج',
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.7),
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                const SizedBox(height: 12),

                Expanded(
                  child: ListView.separated(
                    itemCount: response.top5.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 10),
                    itemBuilder: (context, index) {
                      final item = response.top5[index];
                      final isTop = index == 0;

                      return Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 18, vertical: 14),
                        decoration: BoxDecoration(
                          color: isTop
                              ? const Color(0xFF6C63FF).withValues(alpha: 0.12)
                              : Colors.white.withValues(alpha: 0.04),
                          borderRadius: BorderRadius.circular(14),
                          border: isTop
                              ? Border.all(
                                  color: const Color(0xFF6C63FF)
                                      .withValues(alpha: 0.3),
                                )
                              : null,
                        ),
                        child: Row(
                          children: [
                            // Rank
                            Container(
                              width: 32,
                              height: 32,
                              decoration: BoxDecoration(
                                color: isTop
                                    ? const Color(0xFF6C63FF)
                                    : Colors.white.withValues(alpha: 0.1),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Center(
                                child: Text(
                                  '${index + 1}',
                                  style: TextStyle(
                                    color: isTop
                                        ? Colors.white
                                        : Colors.white60,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 14),

                            // Info
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    item.gloss,
                                    textDirection: TextDirection.rtl,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  if (item.meaning.isNotEmpty)
                                    Text(
                                      item.meaning,
                                      textDirection: TextDirection.rtl,
                                      style: TextStyle(
                                        color:
                                            Colors.white.withValues(alpha: 0.5),
                                        fontSize: 12,
                                      ),
                                    ),
                                ],
                              ),
                            ),

                            // Confidence
                            Text(
                              '${item.confidence.toStringAsFixed(1)}%',
                              style: TextStyle(
                                color: _confidenceColor(item.confidence),
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
                const SizedBox(height: 16),

                // Try again button
                SizedBox(
                  width: double.infinity,
                  height: 54,
                  child: ElevatedButton.icon(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.replay_rounded),
                    label: const Text('ترجمة إشارة أخرى'),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Color _confidenceColor(double conf) {
    if (conf >= 50) return const Color(0xFF00E676);
    if (conf >= 25) return const Color(0xFFFFD740);
    return const Color(0xFFFF5252);
  }
}
