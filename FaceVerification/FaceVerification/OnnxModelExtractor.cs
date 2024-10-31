using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace FaceVerification
{
    //定義抽取人臉特徵向量的類別
    public class OnnxModelExtractor : IDisposable
    {
        private InferenceSession _session;    // 存放抽取人臉特徵向量的模型的變數
        private string _inputName;                  // 存放人臉特徵Metadata的變數

        public OnnxModelExtractor(string modelPath)             //建構函式
        {
            _session = new InferenceSession(modelPath);           //利用人臉辨識模型建立InferenceSession類別的物件, 並存放在_session變數 
            _inputName = _session.InputMetadata.Keys.FirstOrDefault();  // 將人臉特徵的Metadata存放在_inputName變數
        }

        public float[] GetEmbedding(string imagePath)           // 支援取得人臉特徵向量的函式
        {
            using var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath);    // 載入代表人臉的圖片檔案
            var input = PreprocessImage(image);                         // 將圖片資料整理成人臉辨識模型需要的規格
            var inputs = new List<NamedOnnxValue>                 // 將圖片資料與metadata準備成人臉辨識模型能夠處理的List集合
            {
                NamedOnnxValue.CreateFromTensor(_inputName, input)
            };
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);   // 取出128個特徵向量
            var embedding = results.First().AsEnumerable<float>().ToArray();        // 將取出的特徵向量轉型成陣列
            return embedding;                                                                                                   // 傳回陣列的內容
        }

        // 支援將圖片資料整理成人臉辨識模型需要的規格的函式
        private Tensor<float> PreprocessImage(Image<Rgb24> image)
        { 
            image.Mutate(x => x.Resize(160, 160));        // 將讀取到的圖片轉換成人臉辨識模型需要的大小
            var input = new DenseTensor<float>(new[] { 1, 160, 160, 3 });   // 建位代表[batch size, height, width, channels]四個維度的DenseTensor
            //調整RGB色彩內容值至0~1之間的範圍並填入DenseTensor
            for (int y = 0; y < 160; y++)
            {
                for (int x = 0; x < 160; x++)
                {
                    var pixel = image[x, y];
                    input[0, 0, y, x] = pixel.R / 255f;
                    input[0, 1, y, x] = pixel.G / 255f;
                    input[0, 2, y, x] = pixel.B / 255f;
                }
            }
            return input;                       // 傳回調整後的內容
        }

        // 支援自動資源管理的函式
        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
