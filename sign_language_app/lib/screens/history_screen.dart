import 'package:flutter/material.dart';
import '../services/prediction_service.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<Map<String, dynamic>> _history = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    setState(() => _isLoading = true);
    try {
      _history = await PredictionService.getHistory();
    } catch (e) {
      // silently fail
    }
    if (mounted) setState(() => _isLoading = false);
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Text(
                  'سجل الترجمات',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.refresh_rounded,
                      color: Colors.white70),
                  onPressed: _loadHistory,
                ),
              ],
            ),
            const SizedBox(height: 20),

            if (_isLoading)
              const Expanded(
                child: Center(
                  child: CircularProgressIndicator(
                    color: Color(0xFF6C63FF),
                  ),
                ),
              )
            else if (_history.isEmpty)
              Expanded(
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.history_rounded,
                        size: 64,
                        color: Colors.white.withValues(alpha: 0.2),
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'لا توجد ترجمات سابقة',
                        style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.4),
                          fontSize: 16,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'ابدأ بتصوير فيديو لغة الإشارة',
                        style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.3),
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ),
              )
            else
              Expanded(
                child: RefreshIndicator(
                  onRefresh: _loadHistory,
                  color: const Color(0xFF6C63FF),
                  child: ListView.separated(
                    itemCount: _history.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 12),
                    itemBuilder: (context, index) {
                      final item = _history[index];
                      final gloss = item['gloss'] ?? '';
                      final meaning = item['meaning'] ?? '';
                      final confidence = (item['confidence'] ?? 0).toDouble();
                      final createdAt =
                          DateTime.tryParse(item['created_at'] ?? '');

                      return Container(
                        padding: const EdgeInsets.all(18),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.05),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: Colors.white.withValues(alpha: 0.08),
                          ),
                        ),
                        child: Row(
                          children: [
                            // Icon
                            Container(
                              width: 48,
                              height: 48,
                              decoration: BoxDecoration(
                                color: const Color(0xFF6C63FF)
                                    .withValues(alpha: 0.15),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: const Icon(
                                Icons.sign_language_rounded,
                                color: Color(0xFF6C63FF),
                              ),
                            ),
                            const SizedBox(width: 14),

                            // Content
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    gloss,
                                    textDirection: TextDirection.rtl,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.w600,
                                      fontSize: 15,
                                    ),
                                  ),
                                  if (meaning.isNotEmpty)
                                    Text(
                                      meaning,
                                      textDirection: TextDirection.rtl,
                                      style: TextStyle(
                                        color:
                                            Colors.white.withValues(alpha: 0.5),
                                        fontSize: 13,
                                      ),
                                    ),
                                  if (createdAt != null)
                                    Text(
                                      _formatDate(createdAt),
                                      style: TextStyle(
                                        color:
                                            Colors.white.withValues(alpha: 0.3),
                                        fontSize: 11,
                                      ),
                                    ),
                                ],
                              ),
                            ),

                            // Confidence
                            Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 10, vertical: 4),
                              decoration: BoxDecoration(
                                color: _confColor(confidence)
                                    .withValues(alpha: 0.15),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Text(
                                '${confidence.toStringAsFixed(1)}%',
                                style: TextStyle(
                                  color: _confColor(confidence),
                                  fontWeight: FontWeight.bold,
                                  fontSize: 13,
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Color _confColor(double c) {
    if (c >= 50) return const Color(0xFF00E676);
    if (c >= 25) return const Color(0xFFFFD740);
    return const Color(0xFFFF5252);
  }

  String _formatDate(DateTime dt) {
    final local = dt.toLocal();
    return '${local.day}/${local.month}/${local.year} - '
        '${local.hour.toString().padLeft(2, '0')}:'
        '${local.minute.toString().padLeft(2, '0')}';
  }
}
