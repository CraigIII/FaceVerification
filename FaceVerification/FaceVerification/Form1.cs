using Microsoft.ML;
using System.Windows.Forms;
//using Microsoft.ML.Transforms.Image;
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;
namespace FaceVerification
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        // 支援判斷兩個陣列對應的內容值的餘弦相似度Cosine Similarity的函式
        float CosineSimilarity(float[] vector1, float[] vector2)
        {
            float dotProduct = vector1.Zip(vector2, (a, b) => a * b).Sum();
            float magnitude1 = (float)Math.Sqrt(vector1.Sum(a => a * a));
            float magnitude2 = (float)Math.Sqrt(vector2.Sum(b => b * b));
            return dotProduct / (magnitude1 * magnitude2);
        }

        private void btnVerify_Click(object sender, EventArgs e)
        {
            var embeddingExtractor = new OnnxModelExtractor(@"Models/facenet.onnx");            // 載入facenet.onnx人臉辨識模型
            float[] embedding1 = embeddingExtractor.GetEmbedding(@"Faces/測試圖片1.jpg");   // 取出第一張圖片的特徵向量
            float[] embedding2 = embeddingExtractor.GetEmbedding(@"Faces/測試圖片2.jpg");   // 取出第二張圖片的特徵向量

            var similarityScore = CosineSimilarity(embedding1, embedding2);                                       // 計算代表兩張圖片的特徵向量的餘弦相似度
            bool isMatch = similarityScore > 0.7;                                                                                                 // 如果餘弦相似度大於0,7(可視需要調整), 則判定兩張圖片代表同一個人
            Trace.WriteLine($"測試圖片1和測試圖片2的相似程度為{similarityScore:P2}, 是否為同一人:{isMatch}");    // 顯示判定結果
        }
    }
}
