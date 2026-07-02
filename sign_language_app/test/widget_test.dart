import 'package:flutter_test/flutter_test.dart';
import 'package:sign_language_app/main.dart';

void main() {
  testWidgets('App loads', (WidgetTester tester) async {
    await tester.pumpWidget(const SignLanguageApp());
    // Basic smoke test
    expect(find.byType(SignLanguageApp), findsOneWidget);
  });
}
