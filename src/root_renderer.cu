#include <root_renderer.cuh>

#include <TApplication.h>
#include <TCanvas.h>
#include <TPolyMarker3D.h>
#include <TSystem.h>
#include <TTimer.h>
#include <TView.h>
#include <algorithm>
#include <cmath>

RootRenderer::~RootRenderer()
{
    shutdown();
}

void RootRenderer::initialize(const std::string &)
{
    m_initialized = true;
}

void RootRenderer::render(const Particles &, size_t)
{
}

void RootRenderer::shutdown()
{
    m_initialized = false;
}

void RootRenderer::display_frame(const Frame &frame)
{
    if (!m_marker || !m_canvas)
        return;

    auto n = static_cast<Int_t>(frame.x.size());

    m_marker->SetPolyMarker(n, static_cast<Double_t *>(nullptr), 1);

    float xmin = 1e30f, xmax = -1e30f;
    float ymin = 1e30f, ymax = -1e30f;
    float zmin = 1e30f, zmax = -1e30f;

    for (Int_t i = 0; i < n; ++i)
    {
        m_marker->SetPoint(i, frame.x[i], frame.y[i], frame.z[i]);
        xmin = std::min(xmin, frame.x[i]);
        xmax = std::max(xmax, frame.x[i]);
        ymin = std::min(ymin, frame.y[i]);
        ymax = std::max(ymax, frame.y[i]);
        zmin = std::min(zmin, frame.z[i]);
        zmax = std::max(zmax, frame.z[i]);
    }

    float pad = std::max({xmax - xmin, ymax - ymin, zmax - zmin, 1.0f}) * 0.5f;
    float cx = (xmin + xmax) * 0.5f, cy = (ymin + ymax) * 0.5f, cz = (zmin + zmax) * 0.5f;

    auto *view = m_canvas->GetView();
    if (view)
        view->SetRange(cx - pad, cy - pad, cz - pad, cx + pad, cy + pad, cz + pad);

    m_canvas->Modified();
    m_canvas->Update();
}

class FrameConsumerTimer : public TTimer
{
  public:
    FrameConsumerTimer(FrameQueue *queue, RootRenderer *renderer, Long_t ms)
        : TTimer(ms, kFALSE), m_queue(queue), m_renderer(renderer)
    {
    }

    Bool_t Notify() override
    {
        auto frame = m_queue->try_pop();
        if (frame)
            m_renderer->display_frame(*frame);
        else if (m_queue->is_closed())
            gSystem->ExitLoop();

        Reset();
        return kTRUE;
    }

  private:
    FrameQueue *m_queue;
    RootRenderer *m_renderer;
};

void RootRenderer::run_event_loop(FrameQueue &queue)
{
    int argc = 0;
    char *argv[] = {nullptr};
    m_app = new TApplication("NBodySim", &argc, argv);

    m_canvas = new TCanvas("c1", "N-Body Simulation", 800, 600);

    TView *view = TView::CreateView(1);
    view->SetRange(-2, -2, -2, 2, 2, 2);

    m_marker = new TPolyMarker3D();
    m_marker->SetMarkerStyle(20);
    m_marker->SetMarkerSize(1.5);
    m_marker->SetMarkerColor(kBlue);
    m_marker->Draw();

    auto timer = new FrameConsumerTimer(&queue, this, 16);
    timer->TurnOn();
    m_timer = timer;

    m_app->Run();

    if (m_timer)
    {
        m_timer->TurnOff();
        delete m_timer;
        m_timer = nullptr;
    }
}
