import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;

class PredictionResult {
  final int labelId;
  final String gloss;
  final String meaning;
  final double confidence;

  PredictionResult({
    required this.labelId,
    required this.gloss,
    required this.meaning,
    required this.confidence,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      labelId: json['label_id'] as int,
      gloss: json['gloss'] as String,
      meaning: json['meaning'] as String,
      confidence: (json['confidence'] as num).toDouble(),
    );
  }
}

class PredictResponse {
  final PredictionResult topPrediction;
  final List<PredictionResult> top5;

  PredictResponse({required this.topPrediction, required this.top5});

  factory PredictResponse.fromJson(Map<String, dynamic> json) {
    return PredictResponse(
      topPrediction: PredictionResult.fromJson(json['top_prediction']),
      top5: (json['top5'] as List)
          .map((e) => PredictionResult.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }
}

class ApiService {
  static String get _baseUrl {
    // URL for the Hugging Face Space API
    return 'https://ahmed-abdrabo-sign-to-text-api.hf.space';
  }

  /// Send landmark features to the inference API and get predictions.
  static Future<PredictResponse> predict(List<List<double>> features) async {
    final uri = Uri.parse('$_baseUrl/predict');

    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'features': features}),
    );

    if (response.statusCode == 200) {
      return PredictResponse.fromJson(
        jsonDecode(response.body) as Map<String, dynamic>,
      );
    } else {
      throw Exception(
        'Prediction failed (${response.statusCode}): ${response.body}',
      );
    }
  }

  /// Upload a video file to the inference API to extract landmarks and predict.
  static Future<PredictResponse> predictVideo({
    Uint8List? bytes,
    String? path,
    required String fileName,
    bool chatbot = false,
  }) async {
    final uri = Uri.parse('$_baseUrl/predict-video${chatbot ? "?chatbot=true" : ""}');
    final request = http.MultipartRequest('POST', uri);

    if (bytes != null) {
      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          bytes,
          filename: fileName,
        ),
      );
    } else if (path != null) {
      request.files.add(
        await http.MultipartFile.fromPath(
          'file',
          path,
          filename: fileName,
        ),
      );
    } else {
      throw Exception('No video file provided');
    }

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return PredictResponse.fromJson(
        jsonDecode(response.body) as Map<String, dynamic>,
      );
    } else {
      throw Exception(
        'Prediction failed (${response.statusCode}): ${response.body}',
      );
    }
  }

  /// Send chat message and history to the chatbot API
  static Future<String> sendChatMessage(String message, List<Map<String, String>> history) async {
    final uri = Uri.parse('$_baseUrl/chat');
    
    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'message': message,
        'history': history,
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
      return data['reply'] as String;
    } else {
      throw Exception(
        'Chatbot request failed (${response.statusCode}): ${response.body}',
      );
    }
  }

  /// Check if the API server is reachable.
  static Future<bool> healthCheck() async {
    try {
      final uri = Uri.parse('$_baseUrl/health');
      final response = await http.get(uri).timeout(
            const Duration(seconds: 3),
          );
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
