import 'package:supabase_flutter/supabase_flutter.dart';
import 'api_service.dart';

class PredictionService {
  static final _supabase = Supabase.instance.client;

  /// Save a prediction result to Supabase.
  static Future<void> savePrediction(PredictResponse response) async {
    final user = _supabase.auth.currentUser;
    if (user == null) return;

    await _supabase.from('predictions').insert({
      'user_id': user.id,
      'gloss': response.topPrediction.gloss,
      'meaning': response.topPrediction.meaning,
      'confidence': response.topPrediction.confidence,
      'top5': response.top5
          .map((r) => {
                'label_id': r.labelId,
                'gloss': r.gloss,
                'meaning': r.meaning,
                'confidence': r.confidence,
              })
          .toList(),
    });
  }

  /// Get prediction history for the current user.
  static Future<List<Map<String, dynamic>>> getHistory() async {
    final user = _supabase.auth.currentUser;
    if (user == null) return [];

    final data = await _supabase
        .from('predictions')
        .select()
        .eq('user_id', user.id)
        .order('created_at', ascending: false)
        .limit(50);

    return List<Map<String, dynamic>>.from(data);
  }
}
