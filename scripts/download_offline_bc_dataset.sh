data_dir='your-directory-here'

games='AirRaid Amidar Asteroids Atlantis BankHeist BattleZone Berzerk Bowling Boxing Breakout Carnival Centipede ChopperCommand CrazyClimber DemonAttack DoubleDunk ElevatorAction Enduro FishingDerby Freeway Frostbite Gopher Gravitar Hero IceHockey Jamesbond Kangaroo Krull KungFuMaster MontezumaRevenge MsPacman NameThisGame Phoenix Pitfall PrivateEye Qbert RoadRunner Robotank Skiing Solaris SpaceInvaders StarGunner Tennis TimePilot Tutankham UpNDown VideoPinball WizardOfWor YarsRevenge Zaxxon Alien Assault Asterix BeamRider JourneyEscape Pong Pooyan Riverraid Seaquest Venture'
runs='1 2 3 4 5'
files='action observation reward terminal'
for g in ${games[@]}; do
  mkdir -p "${data_dir}/${g}"
  for f in ${files[@]}; do
    for r in ${runs[@]}; do
      c=50
      if [ ! -f "${data_dir}/${g}/${f}_${r}_${c}.gz" ]; then
        if ! gsutil -q stat "gs://atari-replay-datasets/dqn/${g}/${r}/replay_logs/\$store\$_${f}_ckpt.${c}.gz"; then
          c=49
          if ! gsutil -q stat "gs://atari-replay-datasets/dqn/${g}/${r}/replay_logs/\$store\$_${f}_ckpt.${c}.gz"; then
            c=48
            if ! gsutil -q stat "gs://atari-replay-datasets/dqn/${g}/${r}/replay_logs/\$store\$_${f}_ckpt.${c}.gz"; then
              echo "SOMETHING WENT WRONG!"
            fi;
          fi;
        fi;
        gsutil cp "gs://atari-replay-datasets/dqn/${g}/${r}/replay_logs/\$store\$_${f}_ckpt.${c}.gz" "${data_dir}/${g}/${f}_${r}_50.gz"
      fi;
    done;
  done;
done;